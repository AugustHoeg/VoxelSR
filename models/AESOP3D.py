import torch.nn as nn
from copy import deepcopy
from models.RRDBNet3D_official import RRDB, make_layer
from models.RRDBNet3D_official import RRDBNet
from models.models_3D import PixelUnshuffle3D

class AESOP3D(nn.Module):
    """
    AutoEncoder architecture for AESOP loss
    """
    def __init__(self, up_factor, enc_opt, dec_opt):
        super().__init__()

        dec_opt = deepcopy(dec_opt)
        enc_opt = deepcopy(enc_opt)

        # decoder
        self.decoder = RRDBNet(up_factor, dec_opt.in_nc, dec_opt.out_nc, dec_opt.nf, dec_opt.nb, dec_opt.gc, dec_opt.use_checkpoint)  # only implemented for RRDBNet

        # encoder
        self.conv_first = nn.Sequential(
            nn.Conv3d(enc_opt.in_nc, enc_opt.nf // up_factor**3, 3, 1, 1),
            nn.Conv3d(enc_opt.nf // up_factor**3, enc_opt.nf // up_factor**3, 3, 1, 1),
        )
        self.down = []
        if up_factor >= 2:
            self.down.append(PixelUnshuffle3D(2))
        if up_factor >= 4:
            self.down.append(PixelUnshuffle3D(2))
        self.down = nn.Sequential(*self.down)

        self.body = make_layer(RRDB, n_layers=enc_opt.nb, nf=dec_opt.nf, gc=dec_opt.gc)
        self.conv_last = nn.Sequential(
            nn.Conv3d(enc_opt.nf, enc_opt.nf, 3, 1, 1),
            nn.Conv3d(enc_opt.nf, enc_opt.in_nc, 3, 1, 1),
        )

        # misc
        self.dec_is_frozen = False
        self.enc_is_frozen = False
        # default_init_weights([self.conv_first, self.conv_last], 0.1)  # dont re-initiate rrdb weights. already done.
        self.encoder = nn.Sequential(
            self.conv_first,
            self.down,
            self.body,
            self.conv_last,
        )


    def freeze_encoder(self, current_iter="ITER_NOT_GIVEN"):
        if not self.enc_is_frozen:
            self.enc_is_frozen = True
            print(f'Freeze encoder at {current_iter} iterations.')
            for param in self.encoder.parameters():
                param.requires_grad = False

    def freeze_decoder(self, current_iter="ITER_NOT_GIVEN"):
        if not self.dec_is_frozen:
            self.dec_is_frozen = True
            print(f'Freeze decoder at {current_iter} iterations.')
            for param in self.decoder.parameters():
                param.requires_grad = False


    def unfreeze_encoder(self, current_iter="ITER_NOT_GIVEN"):
        if self.enc_is_frozen:
            self.enc_is_frozen = False
            print(f'Unfreeze encoder at {current_iter} iterations.')
            for param in self.encoder.parameters():
                param.requires_grad = True

    def unfreeze_decoder(self, current_iter="ITER_NOT_GIVEN"):
        if self.dec_is_frozen:
            self.dec_is_frozen = False
            print(f'Unfreeze decoder at {current_iter} iterations.')
            for param in self.decoder.parameters():
                param.requires_grad = True



    def forward(self, x, return_bottleneck=True):

        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        if return_bottleneck:
            return x, bottleneck
        else:
            return x
