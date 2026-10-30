import os

import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.POLYP_dataloader1 import POLYP_dataset
from dataloaders.transform import collate_fn_w_transform, collate_fn_wo_transform
from dataloaders.normalize import imagenet_normalize
from networks.ResUnet import Encoder, Decoder, Rec_Decoder
from torch.optim import Adam
from utils.loss import Seg_loss, Rec_loss
from tqdm import tqdm
from utils.tools import seed_torch, setup_logger
from utils.fourier import freq_space_interpolation
from utils.metrics import calculate_metrics
import random

import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Source_Dataset', type=str, nargs='+', default=[ 'CVC-ClinicDB', 'ETIS-LaribPolypDB', 'Kvasir-SEG'], help='BKAI/CVC-ClinicDB/ETIS-LaribPolypDB/Kvasir-SEG')
    parser.add_argument('--Target_Dataset', type=str, nargs='+', default=['BKAI',], help='BKAI/CVC-ClinicDB/ETIS-LaribPolypDB/Kvasir-SEG')
    parser.add_argument('--save_pre_model', type=str, default='./pre_models/')
    parser.add_argument('--dataset_root', type=str, default='./Polyp_Images/')
    # parser.add_argument('--dataset_root', type=str, default='../../mainproject_polyp/Polyp_Images/')
    parser.add_argument('--image_size', type=int, default=352)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--norm', type=str, default='dsgn', help='normalization type')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--mix_layers', type=str, nargs='+', default=['layer1'], help='At which layers is it mixed')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_domains', type=int, default=4, help='number of domains')

    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda')
    else:
        parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args




class Pre_training:
    def __init__(self, args, train_loader, valid_loader=None, lunda_comparative=0.5):
        
        # Load data
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # model
        self.device = args.device
        self.num_domains = self.args.num_domains  # 域数量
        self.encoder = Encoder(resnet='resnet34', mixstyle_layers=self.args.mix_layers, pretrained=True).to(self.device)
        self.decoder = Decoder(feature_channels=[64, 64, 128, 256, 512], num_classes=self.args.num_classes,).to(self.device)
        self.rec_decoder = Rec_Decoder(num_classes=self.args.in_channels, norm=self.args.norm, num_domains=self.num_domains).to(self.device)

        # optimizer
        self.optimizer = Adam([{"params": self.encoder.parameters(), 'lr': self.args.lr / 2},
                           {"params": self.decoder.parameters(), 'lr': self.args.lr},
                           {"params": self.rec_decoder.parameters(), 'lr': self.args.lr}],
                           lr=self.args.lr, betas=(0.9, 0.999))

        # loss
        self.seg_cost = Seg_loss()
        self.rec_cost = Rec_loss()
        self.feat_con = nn.MSELoss()


        # others
        self.num_epochs = self.args.num_epochs
        self.batch_size = self.args.batch_size
        self.model_path = self.args.save_pre_model
        os.makedirs(self.model_path, exist_ok=True)
        self.valid_frequency = 1   # How many rounds of testing are there
        self.lunda_comparative = lunda_comparative
        
        

    def run(self, source_valid_dataloader):
        logger = setup_logger("./logs/train/", log_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        best_valid_dice = 0.0
        domain_map = {'A':0, 'B':1, 'C':2, 'D':3}
        logger.info(self.args)
        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.decoder.train()
            self.rec_decoder.train()
            current_scale = random.choice([-3.0, -1.0, 0.0, 1.0, 3.0])
            self.train_loader.dataset.set_epoch_scale(current_scale)
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.num_epochs}]", ncols=80)
            pre_train_loss = 0.0
            for batch, data in enumerate(pbar):
                x, y, domain_label = data['data'], data['mask'], data['domain']
                x_orig = x.copy()
                x = freq_space_interpolation(x)

                x_orig = torch.from_numpy(imagenet_normalize(x_orig)).to(dtype=torch.float32)
                x = torch.from_numpy(imagenet_normalize(x)).to(dtype=torch.float32)
                y = torch.from_numpy(y).to(dtype=torch.float32)

                x_orig = x_orig.to(self.device)
                x = x.to(self.device)
                y = y.to(self.device)
                

                domain_idx = torch.tensor([domain_map.get(d) for d in domain_label], dtype=torch.long, device=self.device)
                mid_x, sfs = self.encoder(x_orig, is_mix=True)

                mid_x_dis, sfs_dis = self.encoder(x, is_mix=False)
                
                rec_x = self.rec_decoder(mid_x, domain_label=domain_idx)
                mid_rec_x = self.rec_decoder(mid_x_dis, domain_label=domain_idx)

                pred_x = self.decoder(mid_x, sfs)
                pred_x_dis = self.decoder(mid_x_dis, sfs_dis)

                rec_loss = self.rec_cost(rec_x, x_orig) + self.rec_cost(mid_rec_x, x_orig)
                seg_loss = self.seg_cost(pred_x, y) + self.seg_cost(pred_x_dis, y) 
                comparative_loss = self.feat_con(torch.sigmoid(pred_x), torch.sigmoid(pred_x_dis))
                
                loss = (rec_loss + seg_loss + 
                       self.lunda_comparative * comparative_loss )

                
                pre_train_loss += loss.item() 
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})



            total_loss = pre_train_loss / len(self.train_loader)
            logger.info("==================================================================")
            logger.info(
                f"Epoch [{epoch+1}/{self.num_epochs}] | "
                f"Train Loss: {total_loss:.4f} | "
            )

            if  (epoch + 1) % self.valid_frequency == 0:
                self.encoder.eval()
                self.decoder.eval()
                self.rec_decoder.eval()
                print_test_metric = {}
                with torch.no_grad():
                    dice_y = [[], [], []]
                    dice_dict = [f'valid_Dice', f'valid_Enhanced_Align', f'valid_Structure_Measure']
                    for batch, data in enumerate(source_valid_dataloader):
                        x, y = data['data'], data['mask']
                        x = torch.from_numpy(x).to(dtype=torch.float32)
                        y = torch.from_numpy(y).to(dtype=torch.float32)
                        x, y = x.to(self.device), y.to(self.device)
                        mid_x, sfs = self.encoder(x)

                        pred_x = self.decoder(mid_x, sfs)
                        pred = torch.sigmoid(pred_x)
                        
                        metrics = calculate_metrics(pred.detach().cpu(), y.detach().cpu())


                        for i in range(len(metrics)):
                            dice_y[i].append(metrics[i])

                    test_dice_y = np.mean(dice_y, axis=1)
                    for i in range(len(test_dice_y)):
                        print_test_metric[dice_dict[i]] = test_dice_y[i]
                        
                
    
                    
                    best_save_path = os.path.join(self.model_path, f"best_valid_dice.pth")
                    avg_valid_dice = print_test_metric[dice_dict[0]]
                    if avg_valid_dice > best_valid_dice:
                        best_valid_dice = avg_valid_dice

                        torch.save({
                            'encoder_state_dict': self.encoder.state_dict(),
                            'decoder_state_dict': self.decoder.state_dict(),
                            'rec_decoder_state_dict': self.rec_decoder.state_dict(),
                        }, best_save_path) 
                        logger.info(f"The best model has been updated to：{best_save_path}")    
   
        torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'rec_decoder_state_dict': self.rec_decoder.state_dict(),
                }, os.path.join(self.model_path, f"last_model.pth"))                                       


def main(args):
    source_name = args.Source_Dataset 
    source_csv = []
    for name in source_name:
        source_csv.append(name + '_train.csv')
    source_img_list, source_label_list, domain_list = convert_labeled_list(args.dataset_root, source_csv)
    print('Pre_training Phase')
    source_dataset = POLYP_dataset(args.dataset_root, source_img_list, source_label_list, args.image_size,  domain_list=domain_list, img_normalize=False, augment=True)
    source_dataloader = DataLoader(dataset=source_dataset, 
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   collate_fn=collate_fn_w_transform,
                                   num_workers=args.num_workers)
    valid_csv = []
    for name in args.Source_Dataset:
        valid_csv.append(name + '_test.csv')
    source_valid_img_list, source_valid_label_list, source_valid_domain_list = convert_labeled_list(args.dataset_root, valid_csv)
    source_valid_dataset = POLYP_dataset(args.dataset_root, source_valid_img_list, source_valid_label_list, args.image_size,  domain_list=source_valid_domain_list, img_normalize=True, augment=False)
    source_valid_dataloader = DataLoader(dataset=source_valid_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True,
                                  collate_fn=collate_fn_wo_transform,
                                  num_workers=args.num_workers)
    
    pre_train = Pre_training(args, source_dataloader)
    pre_train.run(source_valid_dataloader)
    
    


if __name__ == '__main__':
    seed_torch(42)
    args = parse_args()
    main(args)