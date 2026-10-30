import os
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.PROSTATE_dataloader import PROSTATE_dataset
from dataloaders.transform import collate_fn_w_transform, collate_fn_wo_transform
from dataloaders.normalize import normalize_image
from networks.ResUnet import Encoder, Decoder, Rec_Decoder
from torch.optim import Adam
from utils.loss import Seg_loss, Rec_loss
from tqdm import tqdm
from utils.tools import seed_torch, setup_logger
from torch.optim.lr_scheduler import  ExponentialLR
from utils.fourier import freq_space_interpolation
from utils.metrics import calculate_metrics

import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Source_Dataset', type=str, nargs='+', default=['BMC, HK, I2CVB, RUNMC, UCL'], help='BIDMC/BMC/HK/I2CVB/RUNMC/UCL')
    parser.add_argument('--Target_Dataset', type=str, nargs='+', default=['BIDMC'], help='BIDMC/BMC/HK/I2CVB/RUNMC/UCL')
    parser.add_argument('--save_pre_model', type=str, default='./pre_models/')
    parser.add_argument('--dataset_root', type=str, default='../../mainproject_prostate/Processed_data_ni')
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--norm', type=str, default='dsgn', help='normalization type')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    parser.add_argument('--mix_layers', type=int, nargs='+', default=['layer1'], help='At which layers is it mixed')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_domains', type=int, default=6, help='number of domains')

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
        
        

    def run(self, source_valid_dataloader):
        logger = setup_logger("./logs/train/", log_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        best_valid_dice = 0.0
        logger.info(self.args)
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
                domain_idx = torch.from_numpy(domain_label).to(dtype=torch.long)
                
                x_orig = x_orig.to(self.device)
                x = x.to(self.device)
                y = y.to(self.device)
                domain_idx = domain_idx.to(self.device)
                
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
                       self.lunda_comparative * comparative_loss) 
                
                
                pre_train_loss += loss.item() 
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()
                pbar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})


            total_loss = pre_train_loss / len(self.train_loader)
            logger.info("=======================================================")
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
                    last_name = None
                    dice_y = [[]]
                    dice_dict = [f'Valid_Dice',]
                    for batch, data in enumerate(source_valid_dataloader):
                        x, y, path = data['data'], data['mask'], data['name']
                        current_name = path
                        if last_name is None:
                            last_name = path
                        x = torch.from_numpy(x).to(dtype=torch.float32)
                        y = torch.from_numpy(y).to(dtype=torch.float32)
                        x, y = x.to(self.device), y.to(self.device)
                        mid_x, sfs = self.encoder(x)

                        pred_x = self.decoder(mid_x, sfs)
                        pred = torch.sigmoid(pred_x.detach().cpu())
                        
                        if current_name != last_name:
                            metrics = calculate_metrics(seg_pred3D, y3D)
                            for i in range(len(metrics)):
                                dice_y[i].append(metrics[i])
                            del seg_pred3D
                            del y3D 

                        try:
                            seg_pred3D = torch.cat((pred.unsqueeze(2), seg_pred3D), 2)
                            y3D = torch.cat((y.unsqueeze(2), y3D), 2)
                        except:
                            seg_pred3D = pred.unsqueeze(2)
                            y3D = y.unsqueeze(2) 
                        last_name = current_name

                
                    # Calculate the last 3D volume
                    metrics = calculate_metrics(seg_pred3D, y3D)
                    for i in range(len(metrics)):
                        dice_y[i].append(metrics[i])

                    test_dice_y = np.mean(dice_y, axis=1)
                    for i in range(len(test_dice_y)):
                        print_test_metric[dice_dict[i]] = test_dice_y[i]
                             
                    best_save_path = os.path.join(self.model_path, f"best_valid_dice_.pth")
                    avg_valid_dice = print_test_metric[dice_dict[0]]
                    if avg_valid_dice > best_valid_dice:
                        best_valid_dice = avg_valid_dice

                        torch.save({
                            'encoder_state_dict': self.encoder.state_dict(),
                            'decoder_state_dict': self.decoder.state_dict(),
                            'rec_decoder_state_dict': self.rec_decoder.state_dict(),
                        }, best_save_path) 
                        logger.info(f"The best valid model has been updated to：{best_save_path}") 

            
   
        torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'rec_decoder_state_dict': self.rec_decoder.state_dict(),
                }, os.path.join(self.model_path, f"last_model.pth"))                                       


def main(args):
    source_name = args.Source_Dataset 
    source_csv = []
    for name in source_name:
        source_csv.append(name + '.csv')

    source_img_list, source_label_list, source_domain_label_list = convert_labeled_list(args.dataset_root, source_csv)

    num_samples = len(source_img_list)
    val_size = int(num_samples * 0.2)
    val_size = max(1, val_size) if num_samples > 1 else 0

    all_idx = np.random.permutation(num_samples)
    val_idx = all_idx[:val_size]
    train_idx = all_idx[val_size:]

    source_img_train_list = [source_img_list[i] for i in train_idx]
    source_label_train_list = [source_label_list[i] for i in train_idx]
    source_domain_train_list = [source_domain_label_list[i] for i in train_idx]
    source_img_valid_list = [source_img_list[i] for i in val_idx]
    source_label_valid_list = [source_label_list[i] for i in val_idx]
    source_domain_valid_list = [source_domain_label_list[i] for i in val_idx]
    source_dataset = PROSTATE_dataset(args.dataset_root, source_img_train_list, source_label_train_list, args.image_size, domain_list=source_domain_train_list, img_normalize=False)
    source_dataloader = DataLoader(dataset=source_dataset, 
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   collate_fn=collate_fn_w_transform, 
                                   num_workers=args.num_workers)
    source_valid_dataset = PROSTATE_dataset(args.dataset_root, source_img_valid_list, source_label_valid_list, args.image_size, domain_list=source_domain_valid_list, img_normalize=True)
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