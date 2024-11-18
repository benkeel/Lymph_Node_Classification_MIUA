import torch
import torch.nn as nn
import torch.nn.functional as F


## Definition of VAE model

# Class for convolutional block (for encoder):
# 2d convolution followed by 2d batch normalisation and GELU activation
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU() # check if these need to be swapped to gelu, batchnorm and see if gelu is better than relu etc.
            #nn.LeakyReLU()

        )

    def forward(self, x):
        return self.conv(x)


# Class for convolutional transpose block (for decoder):
# 2d convolutional transpose followed by 2d batch normalisation and GELU activation
class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
            #nn.ReLU()
            #nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

# Class for final convolutional transpose block: changes activation from GELU to Sigmoid. Errors are not symmetric so using a symmetrical activation function helps improve model performance. Tanh will also likely work and can easily change below to test this.
class ConvTransposeFinal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTransposeFinal, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid() # nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

# Sidelined for future work: change convolutional transpose to bilinear upsampling with convolutional layer.
# https://distill.pub/2016/deconv-checkerboard/
# may help to reduce artifacts in reconstruction
class ConvUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvUpsampling, self).__init__()

        self.scale_factor = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), #, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
            #nn.LeakyReLU()
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv(x)

# Main VAE model code (using conv blocks from above)
class VAE_2D(nn.Module):
    def __init__(self, hyperparams):
        super(VAE_2D, self).__init__()

        # Get model hyperparameters
        base = hyperparams['base']
        latent_size = hyperparams['latent_size']

        # output_width = [ (input_width - kernel_width + 2*padding) / stride ] + 1
        self.encoder = nn.Sequential(
            Conv(1, base, 3, stride=1, padding=1),        # (32 - 3 + 2)/1 + 1 = 32
            Conv(base, base, 3, stride=1, padding=1),   # 32
            Conv(base, 2*base, 3, stride=2, padding=1), # (32 - 3 + 2)/2 + 1 = 16
            Conv(2*base, 4*base, 3, stride=1, padding=1), # 16
            Conv(4*base, 4*base, 3, stride=2, padding=1), # (16 - 3 + 2)/2 + 1 = 8
            nn.Conv2d(4*base, 16*base, 8),                # (8 - 8 + 0)/1 + 1 = 1
            #nn.LeakyReLU()
            nn.GELU()
        )
        # change to linear?
        self.encoder_mu = nn.Conv2d(16*base, latent_size*base, 1) # (1 - 1)/1 + 1 = 1    ## 32*base = 32*4 = 128  ### 1*512 = 512
        self.encoder_logvar = nn.Conv2d(16*base, latent_size*base, 1)
        # Note:
        # Conv2d_output_width = [ (input_width - kernel_width + 2*padding) / stride ] + 1
        # ConvTranspose_output_width =(input_width −1)*stride − 2*in_padding + dilation*(kernel_width−1) + out_padding +1
        self.decoder = nn.Sequential(
             nn.Conv2d(latent_size*base, 16*base, 1),                       # (1 - 1)/1 + 1 = 1                              ## 32 64
             ConvTranspose(16*base, 4*base, 8),                    # (1-1)*1 + 2*0 + 1(8-1) + 0 + 1  = 8     ## 64 4
             Conv(4*base, 2*base, 3, padding=1),                   # (8 - 3 + 2)/1 + 1 = 8                          ## 4 4
             ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),# (8-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 16     ## 4 4
             Conv(2*base, base, 3, padding=1),                   # (16 - 3 + 2)/1 + 1 = 16                        ## 4 2
             ConvUpsampling(base, base, 4, stride=2, padding=1),# (16-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 32    ## 2 2
             nn.Conv2d(base, 1, 3, padding=1),                     # 32                                             ## 1 1
             nn.Sigmoid() #nn.Tanh()
        )

        # Function to encode an input and return latent vectors
    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    # Reparameterisation function which takes mean and log variance latent vectors and produces an input to the decoder
    # Reparameterisation trick:
    # mean + (sample from N(0,1))*standard_deviation
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # mean=0 , std=1
        #z = mu + logvar*self.N.sample(mu.shape)
        return mu + eps*std

    # Function to decode/reconstruct a given reparameterised latent vector
    def decode(self, z):
        return self.decoder(z)

    # Forward pass of VAE network, given input return the reconstruction and both corresponding latent vectors
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
