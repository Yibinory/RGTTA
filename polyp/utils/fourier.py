import torch
import numpy as np

def extract_amp_spectrum(trg_img):

    fft_trg_np = np.fft.fft2( trg_img, axes=(-2, -1) )
    amp_target, pha_trg = np.abs(fft_trg_np), np.angle(fft_trg_np)

    return amp_target

def amp_spectrum_swap( amp_local, amp_target, L=0.1 , ratio=0):
    
    a_local = np.fft.fftshift(amp_local, axes=(-2, -1) )
    a_trg = np.fft.fftshift(amp_target, axes=(-2, -1) )

    _, _, h, w = a_local.shape
    b = (  np.floor(np.amin((h,w))*L)  ).astype(int)
    c_h = np.floor(h/2.0).astype(int)
    c_w = np.floor(w/2.0).astype(int)

    h1 = c_h-b
    h2 = c_h+b+1
    w1 = c_w-b
    w2 = c_w+b+1

    a_local[:,:,h1:h2,w1:w2] = a_local[:,:,h1:h2,w1:w2] * ratio + a_trg[:,:,h1:h2,w1:w2] * (1- ratio)
    a_local = np.fft.ifftshift( a_local, axes=(-2, -1) )
    return a_local

def freq_space_interpolation( local_img, L=0.1, alpha=0.1):
    
    local_img_np = local_img 
    batch_size = local_img_np.shape[0]
    # 生成随机排列索引，实现批次内振幅交换
    index = np.random.permutation(batch_size)
    ampl1_reordered = local_img_np[index].copy()  # numpy中用copy()替代detach()的内存隔离效果

    # get fft of local sample
    fft_local_np = np.fft.fft2(local_img_np, axes=(-2, -1) )

    amp_target = extract_amp_spectrum(ampl1_reordered)

    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    lunda = generate_lam_for_statistics(local_img_np, alpha)

    amp_local_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=lunda)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)

    return local_in_trg


def freq_space_interpolation_test( local_img, source_amp, L=0.1, alpha=0.1):
    
    local_img_np = local_img 
    size = len(source_amp)
    
    index = np.random.randint(low=0, high=size)
    amp_target = source_amp[index]

    # get fft of local sample
    fft_local_np = np.fft.fft2(local_img_np, axes=(-2, -1) )


    # extract amplitude and phase of local sample
    amp_local, pha_local = np.abs(fft_local_np), np.angle(fft_local_np)

    # swap the amplitude part of local image with target amplitude spectrum
    lunda = generate_lam_for_statistics(local_img_np, alpha)

    amp_local_ = amp_spectrum_swap( amp_local, amp_target, L=L , ratio=lunda)

    # get transformed image via inverse fft
    fft_local_ = amp_local_ * np.exp( 1j * pha_local )
    local_in_trg = np.fft.ifft2( fft_local_, axes=(-2, -1) )
    local_in_trg = np.real(local_in_trg)

    return local_in_trg

def swap_amp(ampl1):
    batch_size = ampl1.shape[0]
    index = torch.randperm(batch_size, device=ampl1.device)
    ampl1_reordered = ampl1[index].detach()
    
    lambda_ = generate_lam_for_statistics(ampl1)
    ampl_mixed = ampl1 * lambda_ + ampl1_reordered * (1 - lambda_)
    
    return ampl_mixed



def generate_lam_for_statistics(x, alpha=0.1):
    if alpha <= 0:
        raise ValueError(f"alpha must be greater than 0, got {alpha}")
    
    batch_size = x.shape[0]
    channels = x.shape[1]
    lambda_ = np.random.beta(alpha, alpha, size=(batch_size, channels))
    lambda_ = lambda_.reshape(batch_size, channels, 1, 1)
    return lambda_