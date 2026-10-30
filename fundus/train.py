import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.transform import collate_fn_w_transform, collate_fn_wo_transform
from dataloaders.normalize import normalize_image
from networks.ResUnet import Encoder, Decoder, Rec_Decoder
from torch.optim import Adam
from utils.loss import *
from tqdm import tqdm
from utils.tools import seed_torch, setup_logger
from utils.fourier import freq_space_interpolation
from utils.metrics import calculate_metrics

import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Source_Dataset', type=str, nargs='+', default=['REFUGE','ORIGA','REFUGE_Valid', 'Drishti_GS'], help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    parser.add_argument('--Target_Dataset', type=str, nargs='+', default=['RIM_ONE_r3'], help='RIM-ONE-r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    parser.add_argument('--save_pre_model', type=str, default='./pre_models/')
    parser.add_argument('--dataset_root', type=str, default='./Processed_Fundus_Images')
    # parser.add_argument('--dataset_root', type=str, default='../../mainproject_fundus/Processed_Fundus_Images')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--norm', type=str, default='dsgn', help='normalization type')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--mix_layers', type=str, nargs='+', default=['layer1'], help='At which layers is it mixed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_domains', type=int, default=5, help='number of domains')

    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda')
    else:
        parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args




class Pre_training:
    def __init__(self, args, train_loader, source_valid_loader=None, valid_loader=None, lunda_comparative=0.5):
        
        # Load data
        self.args = args
        self.train_loader = train_loader
        self.source_valid_loader = source_valid_loader
        self.valid_loader = valid_loader

        # model
        self.device = args.device
        self.num_domains = self.args.num_domains  
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
        
        

    def run(self):
        logger = setup_logger("./logs/train/", log_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        best_dice = 0.0

        domain_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

        logger.info(args)
        for epoch in range(self.num_epochs):
            self.encoder.train()
            self.decoder.train()
            self.rec_decoder.train()
            pbar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.num_epochs}]", ncols=80)
            pre_train_loss = 0.0
            for batch, data in enumerate(pbar):
                x, y, domain_label = data['data'], data['mask'], data['domain']
                x_orig = x.copy()

                x = freq_space_interpolation(x)

                x_orig = torch.from_numpy(normalize_image(x_orig)).to(dtype=torch.float32)
                x = torch.from_numpy(normalize_image(x)).to(dtype=torch.float32)
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
                f"Train Loss: {total_loss:.4f}  "
            )

            if  (epoch + 1) % self.valid_frequency == 0:
                self.encoder.eval()
                self.decoder.eval()
                self.rec_decoder.eval()
                print_test_metric = {}
                with torch.no_grad():
                    dice_y = [[], []]
                    dice_dict = [f'source_Valid_Disc_Dice', f'source_Valid_Cup_Dice']

                    for batch, data in enumerate(self.source_valid_loader):
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
                        
                
                    best_save_path = os.path.join(self.model_path, f"best_epoch_loss_.pth")
                    avg_dice = (print_test_metric[dice_dict[0]] + print_test_metric[dice_dict[1]]) / 2.0
                    if avg_dice > best_dice:
                        best_dice = avg_dice

                        torch.save({
                            'encoder_state_dict': self.encoder.state_dict(),
                            'decoder_state_dict': self.decoder.state_dict(),
                            'rec_decoder_state_dict': self.rec_decoder.state_dict(),
                        }, best_save_path) 
   
        torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'rec_decoder_state_dict': self.rec_decoder.state_dict(),
                }, os.path.join(self.model_path, f"last_model.pth"))                                       



def main(args):
    source_name = args.Source_Dataset 
    source_csv = []
    target_valid_csv = []
    valid_csv = []
    for name in source_name:
        if name == 'REFUGE_Valid':
            source_csv.append('REFUGE_Valid.csv')
        else:
            source_csv.append(name + '_train.csv')
    source_img_list, source_label_list, domain_list = convert_labeled_list(args.dataset_root, source_csv)
    print('Pre_training Phase')
    source_dataset = OPTIC_dataset(args.dataset_root, source_img_list, source_label_list, args.image_size, img_normalize=False, domain_list=domain_list)
    source_dataloader = DataLoader(dataset=source_dataset, 
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   collate_fn=collate_fn_w_transform, 
                                   num_workers=args.num_workers)
    
    for name in source_name:
        if name == 'REFUGE_Valid':
            continue
        else:
            valid_csv.append(name + '_test.csv')
    source_valid_img_list, source_valid_label_list, source_valid_domain_list = convert_labeled_list(args.dataset_root, valid_csv)
    source_valid_dataset = OPTIC_dataset(args.dataset_root, source_valid_img_list, source_valid_label_list, args.image_size, img_normalize=True, domain_list=None)
    source_valid_dataloader = DataLoader(dataset=source_valid_dataset, 
                                   batch_size=1,
                                   shuffle=False,
                                   pin_memory=True,
                                   collate_fn=collate_fn_wo_transform, 
                                   num_workers=args.num_workers)
    
    for name in args.Target_Dataset:
        if name == 'REFUGE_Valid':
            target_valid_csv.append('REFUGE_Valid.csv')
        else:
            target_valid_csv.append(name + '_test.csv')
            target_valid_csv.append(name + '_train.csv')
    valid_img_list, valid_label_list, valid_domain_list = convert_labeled_list(args.dataset_root,target_valid_csv)
    valid_dataset = OPTIC_dataset(args.dataset_root, valid_img_list, valid_label_list, args.image_size, img_normalize=True, domain_list=None)
    valid_dataloader = DataLoader(dataset=valid_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True,
                                  collate_fn=collate_fn_wo_transform,
                                  num_workers=args.num_workers)
    pre_train = Pre_training(args, source_dataloader, source_valid_dataloader, valid_dataloader)
    pre_train.run()
    
    


if __name__ == '__main__':
    seed_torch(42)
    args = parse_args()
    main(args)