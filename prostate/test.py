import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from utils.tools import seed_torch, setup_logger
from dataloaders.convert_csv_to_list import convert_labeled_list
from dataloaders.PROSTATE_dataloader import PROSTATE_dataset
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
    parser.add_argument('--Target_Dataset', type=str, nargs='+', default=['BIDMC'], help='BIDMC/BMC/HK/I2CVB/RUNMC/UCL')
    parser.add_argument('--dataset_root', type=str, default='../../mainproject_prostate/Processed_data_ni')
    parser.add_argument('--image_size', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--in_channels', type=int, default=3, help='number of input channels')
    parser.add_argument('--norm', type=str, default='dsgn', help='normalization type')
    parser.add_argument('--num_classes', type=int, default=1, help='number of classes')
    # parser.add_argument('--mix_layers', type=int, nargs='+', default=['layer1'], help='At which layers is it mixed')
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
                if np in ['weight', 'bias'] :  # weight is scale, bias is shift
                    # print(nm)
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

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

def prob_entropy(prob, eps=1e-6):
    p = prob.clamp(min=eps, max=1.0 - eps)
    return -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p)).mean()


def save_slice_and_mask(image_tensor, mask_tensor, gt_tensor, save_root, img_name):
    image_dir = os.path.join(save_root, 'image')
    mask_dir = os.path.join(save_root, 'mask')
    gt_dir = os.path.join(save_root, 'true_mask')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    image_np = image_tensor.detach().cpu().numpy()[0, 0].astype(np.float32)
    image_np = image_np - image_np.min()
    if image_np.max() > 0:
        image_np = image_np / image_np.max()
    image_np = (image_np * 255).astype(np.uint8)

    mask_np = mask_tensor.detach().cpu().numpy()[0, 0]
    mask_np = (mask_np > 0.5).astype(np.uint8) * 255

    gt_np = gt_tensor.detach().cpu().numpy()[0, 0]
    gt_np = (gt_np > 0.5).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(image_dir, img_name + '_image.png'), image_np)
    cv2.imwrite(os.path.join(mask_dir, img_name + '_mask.png'), mask_np)
    cv2.imwrite(os.path.join(gt_dir, img_name + '_true_mask.png'), gt_np)

def domain_hypothesis_ensemble(
    encoder,
    decoder,
    rec_decoder,
    x,
    num_domains=1,
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
    domain_map = {
        'RUNMC': 0,
        'BMC': 1,
        'I2CVB': 2,
        'UCL': 3,
        'BIDMC': 4,
        'HK': 5
    }
    for name in target_name:
        target_csv.append(name + '.csv')
    target_img_list, target_label_list, _ = convert_labeled_list(args.dataset_root, target_csv)
    target_dataset = PROSTATE_dataset(args.dataset_root, target_img_list, target_label_list, args.image_size, img_normalize=True)
    target_dataloader = DataLoader(dataset=target_dataset, 
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   pin_memory=True,
                                   collate_fn=collate_fn_wo_transform,
                                   num_workers=args.num_workers)
    
    logger.info(args)
    logger.info('Test Phase')
    encoder, decoder, rec_decoder = None, None, None
    if args.pre_train_model:
        encoder = Encoder(resnet='resnet34', pretrained=True).to(args.device)
        decoder = Decoder(num_classes=args.num_classes).to(args.device)
        rec_decoder = Rec_Decoder(num_classes=args.in_channels, norm='dsgn', num_domains=6).to(args.device)
        pretrained_path = args.pre_model
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
    frozen_mask1 = {}
    for name, param in seg_encoder.named_parameters():
        frozen_mask1[name] = not param.requires_grad
    frozen_mask2 = {}
    for name, param in seg_decoder.named_parameters():
        frozen_mask2[name] = not param.requires_grad

    dice_y = [[]]
    dice_dict = [f'{target_name[0]}_Dice']
    last_name = None
    metrics = None
    seg_encoder.eval()
    seg_decoder.eval()
    rec_decoder_cp.eval()
    with torch.no_grad():
        for indx, data in enumerate(target_dataloader):

            x, y, path = data['data'], data['mask'], data['name']
            current_name = path
            if last_name is None:
                last_name = path
            x = torch.from_numpy(x).to(dtype=torch.float32)
            y = torch.from_numpy(y).to(dtype=torch.float32)
            x, y = x.to(args.device), y.to(args.device)
            base_pred = predict_with_flip_tta(seg_encoder, seg_decoder, x)

            seg_pred, ens_weight = domain_hypothesis_ensemble(
                seg_encoder,
                seg_decoder,
                rec_decoder_cp,
                x,
                num_domains=6,
                agree_alpha=args.ens_agree_alpha,
                recon_alpha=args.ens_recon_alpha,
                recon_edge_lambda=args.ens_recon_edge_lambda,
                target_index=domain_map[target_name[0]]
            )
            recon_support = (1.0 - ens_weight[0]).clamp(0.0, 1.0)
            if recon_support.item() > 0.50:
                final_pred = (1.0 - recon_support) * base_pred + recon_support * seg_pred   
            else:
                final_pred = base_pred 

            final_pred = final_pred.detach().cpu()
            if current_name != last_name:   # Calculate the previous 3D volume
                metrics = calculate_metrics(seg_pred3D, y3D)
                for i in range(len(metrics)):
                    dice_y[i].append(metrics[i])

                del seg_pred3D
                del y3D

            try:
                seg_pred3D = torch.cat((final_pred.unsqueeze(2), seg_pred3D), 2)
                y3D = torch.cat((y.unsqueeze(2), y3D), 2)
            except:
                seg_pred3D = final_pred.unsqueeze(2)
                y3D = y.unsqueeze(2)               
            last_name = current_name

            if args.save_results and indx < 300:
                img_name = path[0].split('/')[-1] + '-' + str(y3D.shape[2])
                save_slice_and_mask(x, final_pred, y, args.path_save_result, img_name)
            
        metrics = calculate_metrics(seg_pred3D, y3D)
        for i in range(len(metrics)):
            dice_y[i].append(metrics[i])

        test_dice_y = np.mean(dice_y, axis=1)
        print_test_metric = {}
        for i in range(len(test_dice_y)):
            print_test_metric[dice_dict[i]] = test_dice_y[i]

        
        logger.info("Test Metrics: {}".format(print_test_metric))


if __name__ == '__main__':
    seed_torch(42)
    args = parse_args2()
    main(args)
