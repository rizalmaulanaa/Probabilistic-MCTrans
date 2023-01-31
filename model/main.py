import torch.nn as nn

from centers.center import MCTrans
from encoders.encoder import ResNet
from heads.head import MCTransAuxHead
from decoders.decoder import UNetDecoder
from probabilistic import AxisAlignedConvGaussian, Fcomb


class EncoderDecoder(nn.Module):
    def __init__(self, num_classes=1, d_model=128, num_filters=[64,64,128,128,128], n_points=4, prob=False, latent_dim=6):
        super(EncoderDecoder, self).__init__()
        self.prob = prob
        self.encoder = ResNet(in_channels=1, depth=18, out_indices=(0, 1, 2, 3, 4))
        self.decoder = UNetDecoder(in_channels=num_filters)

        self.center = MCTrans(d_model=d_model, nhead=8, d_ffn=512, dropout=0.1, 
                              act="relu", n_levels=3, n_points=n_points, n_sa_layers=6)
        
        self.head = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        self.aux_head = MCTransAuxHead(d_model=d_model, d_ffn=512, act="relu", 
                                       num_classes=num_classes, in_channles=num_filters)

        if self.prob:
            self.initializers = {'w':'he_normal', 'b':'normal'}
            self.prior = AxisAlignedConvGaussian(input_channels=1, num_filters=num_filters[:-1], 
                                                no_convs_per_block=2, latent_dim=latent_dim,
                                                 initializers=self.initializers)
            self.posterior = AxisAlignedConvGaussian(input_channels=1, num_filters=num_filters[:-1],
                                                    no_convs_per_block=2, latent_dim=latent_dim,
                                                    initializers=self.initializers, posterior=True)
            self.fcomb = Fcomb(num_filters[:-1], latent_dim, num_classes, num_classes, 2, self.initializers, use_tile=True)

    def forward(self, img, seg):
        # Traning
        # MCTrans
        x = self.extract_feat(img)
        x_ = self.decoder(x)
        x_ = self.head(x_)
        logits = self.aux_head(x)
        
        # Probabilisic model
        if self.prob:
            self.posterior_latent_space = self.posterior(img, seg)
            self.prior_latent_space = self.prior(img)
            x_ = self.fcomb(x_, self.posterior_latent_space.rsample())

        return x_, logits

    def sampling(self, img, seg):
        # Inference
        ## MCTrans
        x = self.extract_feat(img)
        x_ = self.decoder(x)
        x_ = self.head(x_)
        logits = self.aux_head(x)
        
        ## Probabilistic model
        if self.prob:
            self.posterior_latent_space = self.posterior(img, seg)
            self.prior_latent_space = self.prior(img)
            x_ = self.fcomb(x_, self.prior_latent_space.sample())

        return x_, logits

    def extract_feat(self, img):
        # Encoder
        x = self.encoder(img)
        x = self.center(x)
        return x

    def kl_divergence(self, analytic=False, calculate_posterior=True, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            # Need to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl._kl_independent_independent(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div