import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, epoch, epochs, hyperparams):
    '''
    Function to calcuate loss between a batch of input data and VAE reconstructions

    Input:
    - recon_x: reconstructed data
    - x: input data
    - mu: mean latent vector
    - logvar: log variance latent vector
    - epoch: current epoch id (used for annealing)
    - hyperparams: dictionary of hyperparameters of model (several influence loss function)

    Output:
    - beta_vae_loss: loss which has been combined with other metrics/scaled
    - recon_loss: L1 loss (mean absolute error)
    - kld: KL divergence
    - ssim_score: SSIM score (structural similarity)
    - pure_loss: un-scaled/altered loss
    '''
    # get hyperparameters
    base = hyperparams["base"]
    latent_size = hyperparams["latent_size"]
    annealing = hyperparams["annealing"]
    ssim_indicator = hyperparams["ssim_indicator"]
    alpha = hyperparams["alpha"]
    beta = hyperparams["beta"]
    batch_size = hyperparams["batch_size"]
    ssim_scalar = hyperparams["ssim_scalar"]
    recon_scale_factor = hyperparams["recon_scale_factor"]

    # linear annealing: exponentially increase the effect of KL divergence over time
    if annealing == 1:
        annealing_weight = kl_annealing(epoch, epochs, 1) # was 3 before!!!
    if annealing == 0:
        annealing_weight = 1

    # KL Divergence = -0.5 * sum(1 + log(variance) - mu^2 - exponential(logvar))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_mean = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1))
    #kld = kld_mean * scale_factor

    l1_loss = nn.L1Loss(reduction='mean') #sum  # could be reduction mean
    recon_loss = l1_loss(recon_x, x) * recon_scale_factor

    # reduction: mean - may get a most consistent quality across the dataset
    l1_loss_mean = nn.L1Loss(reduction='mean')
    recon_loss_mean = l1_loss_mean(recon_x, x)

    # upweight SSIM by s.f. relative to batch_size
    ssim_scalar = batch_size*ssim_scalar

    # Choose whether to include SSIM / (other metric if replaced for time series metric)
    # ssim_indicator = 0 if not including
    if ssim_indicator == 0:
        recon_mix = recon_loss
        ssim_loss = torch.tensor(0)

    # ssim_indicator = 1 if including, and weight by alpha / (1-alpha)
    # justification for including in: https://arxiv.org/pdf/1511.08861
    if ssim_indicator == 1:
        ssim_module = SSIM(data_range=1, win_size=5, win_sigma=1.5, size_average=True, channel=1) # win_size is gaussian kernel size, try 3
        #ms_ssim_module = MS_SSIM(data_range=1, win_size=5, win_sigma=1.5, size_average=True, channel=1)
        ssim_loss = 1 - ssim_module(x, recon_x)
        #ssim_loss = 1 - ms_ssim_module(x, recon_x)
        recon_mix = alpha*recon_loss + (1-alpha)*ssim_loss*ssim_scalar


    # idea from beta vae: https://openreview.net/pdf?id=Sy2fzU9gl
    beta_norm = (beta*latent_size*base)/(32*32)
    beta_annealed_kld = beta_norm*kld_mean*annealing_weight
    beta_vae_loss = recon_mix + beta_annealed_kld   #(kld - C).abs()
    pure_loss = recon_loss_mean + kld_mean
    alpha_recon = alpha*recon_loss
    alpha_SSIM = (1-alpha)*ssim_loss*ssim_scalar

    ssim_score = ssim(x, recon_x, data_range=1, nonnegative_ssim=True)#

    return pure_loss, beta_vae_loss, recon_loss_mean, alpha_recon, kld_mean, beta_annealed_kld, ssim_score, alpha_SSIM


def kl_annealing(epoch, max_epochs, weight_max):
    # Compute the annealing weight using (1 - exponential decay function)
    weight = weight_max * (1 - np.exp(-5 * epoch / max_epochs))
    weight = weight + 0.25
    return weight