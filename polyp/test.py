import os
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse, random
import numpy as np
from utils.tools import seed_torch, setup_logger
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.POLYP_dataloader1 import POLYP_dataset
from torch.utils.data import DataLoader

from networks.ResUnet import Encoder, Decoder, Rec_Decoder
from dataloaders.transform import collate_fn_wo_transform
from utils.metrics import calculate_metrics
import cv2
import os
import copy
from datetime import datetime


def parse_args2():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Target_Dataset', type=str, nargs='+', default=['BKAI',], help='BKAI/CVC-ClinicDB/ETIS-LaribPolypDB/Kvasir-SEG')
    parser.add_argument('--dataset_root', type=str, default='../../mainproject_polyp/Polyp_Images/')
    parser.add_argument('--image_size', type=int, default=352)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--norm', type=str, default='dsgn', help='normalization type')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    # parser.add_argument('--mix_layers', type=str, nargs='+', default=['layer1'], help='At which layers is it mixed')
    parser.add_argument('--save_results', action='store_true', default=False, help='Whether to save the test results')
    parser.add_argument('--path_save_result', type=str, default='./test_results/')
    parser.add_argument('--pre_train_model', action='store_true', default=False, help='Whether to use pre-trained model')
    parser.add_argument('--pre_model', type=str, default='./pre_models/last_model.pth')
    parser.add_argument('--ens_agree_alpha', type=float, default=3.0, help='Penalty strength for disagreement with base hypothesis')
    parser.add_argument('--ens_recon_alpha', type=float, default=2.0, help='Weight for reconstruction-fidelity term in ensemble scoring')
    parser.add_argument('--ens_recon_edge_lambda', type=float, default=0.5, help='Edge-consistency weight inside reconstruction score')
    
    if torch.cuda.is_available():
        parser.add_argument('--device', type=str, default='cuda')
    else:
        parser.add_argument('--device', type=str, default='cpu')

    args = parser.parse_args()
    return args

def predict_with_flip_tta(encoder, decoder, x):
    probs = []
    with torch.no_grad():
        mid_x, sfs = encoder(x)
        logits = decoder(mid_x, sfs)
        probs.append(torch.sigmoid(logits))

        x_flip = torch.flip(x, dims=[-1])
        mid_f, sfs_f = encoder(x_flip)
        logits_f = decoder(mid_f, sfs_f)
        probs.append(torch.flip(torch.sigmoid(logits_f), dims=[-1]))

    return torch.stack(probs, dim=0).mean(dim=0)

def _image_gradients(x):
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return gx, gy


def reconstruction_score(x, rec_x, edge_lambda=0.5):
    pix_l1 = F.l1_loss(rec_x, x)
    gx, gy = _image_gradients(x)
    rgx, rgy = _image_gradients(rec_x)
    grad_l1 = F.l1_loss(rgx, gx) + F.l1_loss(rgy, gy)
    return -(pix_l1 + edge_lambda * grad_l1)

def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def prob_entropy(prob, eps=1e-6):
    p = prob.clamp(min=eps, max=1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()


def domain_hypothesis_ensemble(
    encoder,
    decoder,
    rec_decoder,
    x,
    num_domains=4,
    agree_alpha=3.0,
    recon_alpha=2.0,
    recon_edge_lambda=0.5,
    target_index=0
):
    encoder.eval()
    decoder.eval()
    rec_decoder.eval()

    base_prob = predict_with_flip_tta(encoder, decoder, x)
    hypotheses = [base_prob]
    scores = [-prob_entropy(base_prob)]
    with torch.no_grad():
        mid_x, _ = encoder(x)
        for d in range(num_domains):
            if d == target_index:
                continue
            domain_idx = torch.full((x.size(0),), d, dtype=torch.long, device=x.device)
            rec_x = rec_decoder(mid_x, domain_label=domain_idx)
            p_d = predict_with_flip_tta(encoder, decoder, rec_x)

            ent_d = prob_entropy(p_d)
            agree_d = F.mse_loss(p_d, base_prob)
            recon_d = reconstruction_score(x, rec_x, edge_lambda=recon_edge_lambda)
            score_d = recon_alpha * recon_d - ent_d - agree_alpha * agree_d

            hypotheses.append(p_d)
            scores.append(score_d)

    score_tensor = torch.stack(scores, dim=0)
    weight = torch.softmax(score_tensor, dim=0)

    fused = 0.0
    for i, p in enumerate(hypotheses):
        fused = fused + weight[i] * p

    return fused, weight


def main(args):
    logger = setup_logger("./logs/test/", log_name=f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    target_name = args.Target_Dataset
    target_csv = []
    os.makedirs(args.path_save_result, exist_ok=True)
    for name in target_name:
        target_csv.append(name + '_test.csv')
        target_csv.append(name + '_train.csv')
    target_img_list, target_label_list, _ = convert_labeled_list(args.dataset_root, target_csv)
    target_dataset = POLYP_dataset(args.dataset_root, target_img_list, target_label_list, args.image_size, img_normalize=True)
    target_dataloader = DataLoader(dataset=target_dataset, 
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   collate_fn=collate_fn_wo_transform,
                                   num_workers=args.num_workers)
    
    encoder, decoder, rec_decoder = None, None, None
    if args.pre_train_model:
        pretrained_path = args.pre_model
        encoder = Encoder(resnet='resnet34', pretrained=True, mixstyle_layers=['layer1']).to(args.device)
        decoder = Decoder(num_classes=args.num_classes).to(args.device)
        rec_decoder = Rec_Decoder(num_classes=args.in_channels, norm=args.norm, num_domains=4).to(args.device)
        checkpoint = torch.load(pretrained_path, map_location=torch.device(args.device), weights_only=False) 
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        rec_decoder.load_state_dict(checkpoint['rec_decoder_state_dict']) 
        for param in encoder.parameters():
            param.requires_grad = False     
        for param in decoder.parameters():
            param.requires_grad = False 
        for param in rec_decoder.parameters():
            param.requires_grad = False


    seg_encoder = copy.deepcopy(encoder)
    seg_decoder = copy.deepcopy(decoder)
    rec_decoder_cp = copy.deepcopy(rec_decoder)
 
    for param in seg_encoder.parameters():
       param.requires_grad = False
    for param in seg_decoder.parameters():
        param.requires_grad = False
    for param in rec_decoder_cp.parameters():
        param.requires_grad = False
    


    # BN 
    encoder_parameters, _ = collect_params(seg_encoder)
    decoder_parameters, _ = collect_params(seg_decoder)
    for param in encoder_parameters:
        param.requires_grad = True    
    for param in decoder_parameters:
        param.requires_grad = True    


    domain_map = {
        'BKAI': 0,
        'CVC-ClinicDB': 1,
        'ETIS-LaribPolypDB': 2,
        'Kvasir-SEG': 3
    }
    dice_y = [[], [], []]
    dice_dict = [f'{target_name[0]}_Dice', f'{target_name[0]}_Enhanced_Align', f'{target_name[0]}_Structure_Measure']
    seg_encoder.eval()
    seg_decoder.eval()
    rec_decoder_cp.eval()
    for indx, data in enumerate(target_dataloader):

        x, y, path = data['data'], data['mask'], data['name']
        x = torch.from_numpy(x).to(dtype=torch.float32)
        y = torch.from_numpy(y).to(dtype=torch.float32)
        x, y = x.to(args.device), y.to(args.device)
        with torch.no_grad():
            base_pred = predict_with_flip_tta(seg_encoder, seg_decoder, x)

            seg_pred, ens_weight = domain_hypothesis_ensemble(
                seg_encoder,
                seg_decoder,
                rec_decoder_cp,
                x,
                num_domains=4,
                agree_alpha=args.ens_agree_alpha,
                recon_alpha=args.ens_recon_alpha,
                recon_edge_lambda=args.ens_recon_edge_lambda,
                target_index=domain_map[target_name[0]]
            )  

            recon_support = (1.0 - ens_weight[0]).clamp(0.0, 1.0)
            if recon_support.item() > 0.5:
                final_pred = (1.0 - recon_support) * base_pred + recon_support * seg_pred   
            else:
                final_pred = base_pred    
            metrics = calculate_metrics(final_pred.detach().cpu(), y.detach().cpu())

            for i in range(len(metrics)):
                dice_y[i].append(metrics[i])
            
            if args.save_results and indx < 100:
                draw_output = final_pred.detach().cpu().numpy()
                mask = (draw_output[0][0] > 0.5).astype(np.uint8) * 255
                img_name = path[0].split('/')[-4] + '-' + path[0].split('/')[-3] + path[0].split('/')[-1].split('.')[0]

                if draw_output.shape[1] == 1:
                    cv2.imwrite(args.path_save_result + '/' + img_name + '_Mask.png', mask)
            
    test_dice_y = np.mean(dice_y, axis=1)
    print_test_metric = {}
    for i in range(len(test_dice_y)):
        print_test_metric[dice_dict[i]] = test_dice_y[i]
    
    logger.info("Test Metrics: {}".format(print_test_metric))


if __name__ == '__main__':
    seed_torch(42)
    args = parse_args2()
    main(args)
