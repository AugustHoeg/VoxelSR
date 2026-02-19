import functools
import torch
import torch.nn as nn
from torch.nn import init

"""
# --------------------------------------------
# select the network of G, D and F
# --------------------------------------------
"""

class DummyNetwork(nn.Module):
    def __init__(self):
        super(DummyNetwork, self).__init__()

        self.conv = nn.Conv3d(1, 1, 1, 1, 1)  # dummy conv layer

    def forward(self, x):
        return self.conv(x)

# --------------------------------------------
# Generator, netG, G
# --------------------------------------------
def define_G(opt, mode='train'):
    opt_net = opt['model_opt']['netG']
    model_architecture = opt['model_opt']['model_architecture']

    print("Flash Attention:", torch.backends.cuda.flash_sdp_enabled())

    if model_architecture == "DUMMY":
        netG = DummyNetwork()

    if model_architecture == "DegradeNet":  # DegradeNet
        from models.DegradeNet import DegradeNet as net
        netG = net(down_factor=opt['down_factor'],
                   in_channels=opt_net['in_channels'],
                   out_channels=opt_net['out_channels'],
                   num_feats=opt_net['num_feats'],
                   use_checkpoint=opt_net['use_checkpoint'])

    elif model_architecture == "FlashDegradeNet":  # FlashDegradeNet
        from models.DegradeNet import FlashDegradeNet as net
        netG = net(input_size=opt['dataset_opt']['patch_size_hr'],
                   down_factor=opt['down_factor'],
                   num_blks=opt_net['num_blks'],
                   blk_layers=opt_net['blk_layers'],
                   in_chans=opt_net['in_channels'],
                   shallow_feat=opt_net['shallow_feat'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   attn_window_size=opt_net['attn_window_size'],
                   patch_size=opt_net['patch_size'],
                   skip_dims=opt_net['skip_dims'],
                   drop_path_rate=0.1 if mode == 'train' else 0.0,
                   use_checkpoint=opt_net['use_checkpoint'],
                   upsample_method=opt_net['upsample_method'],
                   requires_grad=True)

    # ----------------------------------------
    # RCAN from paper https://arxiv.org/abs/1807.02758
    # ----------------------------------------
    elif model_architecture == ("RCAN" or "RCAN2d"):  # RCAN
        from models.rcan_arch import RCAN as net
        netG = net(upscale=opt['up_factor'],
                   num_in_ch=opt_net['in_channels'],
                   num_out_ch=opt_net['in_channels'],
                   num_group=opt_net['num_group'],
                   num_block=opt_net['num_block'],
                   num_feat=opt_net['num_feat'],
                   squeeze_factor=opt_net['squeeze_factor'],
                   res_scale=opt_net['res_scale'],
                   rgb_mean=opt_net['rgb_mean'])

    # ----------------------------------------
    # SwinIR from paper https://arxiv.org/pdf/2108.10257
    # ----------------------------------------
    elif model_architecture == ("SwinIR" or "SwinIR2d"):  # SwinIR
        from models.network_swinir import SwinIR as net
        netG = net(upscale=opt['up_factor'],
                   in_chans=opt_net['in_channels'],
                   img_size=opt['dataset_opt']['patch_size'],
                   window_size=opt_net['window_size'],
                   img_range=1.,
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsample_method'],
                   resi_connection='1conv',
                   drop_path_rate=0.1 if mode == 'train' else 0.0,
                   use_checkpoint=opt_net['use_checkpoint'])

    # ----------------------------------------
    # HAT (XPixelGroup) from paper https://arxiv.org/abs/2205.04437
    # ----------------------------------------
    elif model_architecture == ("HAT" or "HAT2d"):  # HAT
        from models.hat_arch import HAT as net
        netG = net(upscale=opt['up_factor'],
                   in_chans=opt_net['in_channels'],
                   img_size=opt['dataset_opt']['patch_size'],
                   window_size=opt_net['window_size'],
                   compress_ratio=opt_net['compress_ratio'],
                   squeeze_factor=opt_net['squeeze_factor'],
                   conv_scale=opt_net['conv_scale'],
                   overlap_ratio=opt_net['overlap_ratio'],
                   img_range=1.,
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsample_method'],
                   resi_connection='1conv',
                   drop_path_rate=0.1 if mode == 'train' else 0.0,
                   use_checkpoint=opt_net['use_checkpoint'])

    # ----------------------------------------
    # DRCT (Dense-Residual-Connected Transformer) from paper: https://arxiv.org/abs/2404.00722
    # ----------------------------------------
    elif model_architecture == ("DRCT" or "DRCT2d"):  # DRCT
        from models.DRCT_arch import DRCT as net
        netG = net(upscale=opt['up_factor'],
                   in_chans=opt_net['in_channels'],
                   img_size=opt['dataset_opt']['patch_size'],
                   window_size=opt_net['window_size'],
                   compress_ratio=opt_net['compress_ratio'],
                   squeeze_factor=opt_net['squeeze_factor'],
                   conv_scale=opt_net['conv_scale'],
                   overlap_ratio=opt_net['overlap_ratio'],
                   img_range=1.,
                   depths=opt_net['depths'],
                   embed_dim=opt_net['embed_dim'],
                   num_heads=opt_net['num_heads'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   upsampler=opt_net['upsample_method'],
                   resi_connection='1conv',
                   drop_path_rate=0.1 if mode == 'train' else 0.0,
                   use_checkpoint=opt_net['use_checkpoint'])


    # ----------------------------------------
    # mDCSRN Generator
    # ----------------------------------------
    elif model_architecture == ("mDCSRN" or "mDCSRN-GAN"):  # mDCSRN
        from models.mDCSRN_GAN import MultiLevelDenseNet as net
        netG = net(up_factor=opt['up_factor'],
                   in_c=opt_net['in_channels'],
                   k_factor=opt_net['k_factor'],
                   k_size=opt_net['k_size'],
                   num_dense_blocks=opt_net['num_dense_blocks'],
                   num_dense_units=opt_net['num_dense_units'],
                   upsample_method=opt_net['upsample_method'],
                   use_checkpoint=opt_net['use_checkpoint'])

    # ----------------------------------------
    # RRDBNet3D / ESRGAN3D Generator
    # ----------------------------------------
    elif model_architecture == ("RRDBNet3D" or "ESRGAN3D"):
        from models.RRDBNet3D_official import RRDBNet as net
        netG = net(up_factor=opt['up_factor'],
                   in_nc=opt_net['in_channels'],
                   out_nc=1,
                   nf=opt_net['num_channels'],
                   nb=opt_net['num_blocks'],
                   gc=32)

    # ----------------------------------------
    # EDDSR
    # ----------------------------------------
    elif model_architecture == ("EDDSR"):
        from models.EDDSR import EDDSR_xs as net
        netG = net(up_factor=opt['up_factor'])


    # ----------------------------------------
    # MFER
    # ----------------------------------------
    elif model_architecture == ("MFER"):
        from models.MFER_official import MFER_xs as net
        netG = net(up_factor=opt['up_factor'])

    # ----------------------------------------
    # 3D Med SwinIR - SuperFormer
    # ----------------------------------------
    elif model_architecture == 'SuperFormer':
        from models.SuperFormer import SuperFormer as net

        netG = net(img_size=opt['dataset_opt']['patch_size'],
                   patch_size=opt_net['patch_size'],
                   in_chans=opt['dataset_opt']['n_channels'],
                   embed_dim=opt_net['embed_dim'],
                   depths=opt_net['depths'],
                   num_heads=opt_net['num_heads'],
                   window_size=opt_net['window_size'],
                   mlp_ratio=opt_net['mlp_ratio'],
                   qkv_bias=True,
                   drop_path_rate=0.1 if mode == 'train' else 0.0,
                   ape=opt_net['ape'],
                   rpb=opt_net['rpb'],
                   patch_norm=True,
                   use_checkpoint=opt_net['use_checkpoint'],
                   upscale=opt['up_factor'],
                   img_range=1.,
                   upsampler=opt_net['upsampler'],
                   resi_connection='1conv',
                   output_type=opt_net['output_type'],
                   num_feat=opt_net['num_feat'],
                   init_type="default")

    # ----------------------------------------
    # MTVNet
    # ----------------------------------------

    elif model_architecture == 'MTVNet':

        H = opt['dataset_opt']['patch_size']
        W = opt['dataset_opt']['patch_size']
        D = opt['dataset_opt']['patch_size']
        input_size = (H, W, D)

        if 'use_monai' in opt_net and opt_net['use_monai']:
            print("Using MTVNet monai version")
            from models.MTVNet_arch_monai import MTVNet_monai as net
            netG = net(input_size=input_size,
                       up_factor=opt['up_factor'],
                       num_levels=opt_net['num_levels'],
                       context_sizes=opt_net['context_sizes'],
                       num_blks=opt_net['num_blks'],
                       blk_layers=opt_net['blk_layers'],
                       in_chans=opt_net['in_channels'],
                       shallow_feats=opt_net['shallow_feats'],
                       pre_up_feats=opt_net['pre_up_feats'],
                       embed_dims=opt_net['embed_dims'],
                       num_heads=opt_net['num_heads'],
                       attn_window_sizes=opt_net['attn_window_sizes'],
                       patch_sizes=opt_net['patch_sizes'],
                       skip_dims=opt_net['skip_dims'],
                       mlp_ratio=opt_net['mlp_ratio'],
                       drop_path_rate=opt_net['drop_path'] if mode == 'train' else 0.0,
                       use_checkpoint=opt_net['use_checkpoint'],
                       upsample_method=opt_net['upsample_method'])

        elif 'use_convnext' in opt_net and opt_net['use_convnext']:
            print("Using MTVNet ConvNext version")
            from models.MTVNeXt_arch_crop_merge import MTVNeXt as net
            netG = net(input_size=input_size,
                       up_factor=opt['up_factor'],
                       num_levels=opt_net['num_levels'],
                       context_sizes=opt_net['context_sizes'],
                       num_blks=opt_net['num_blks'],
                       blk_layers=opt_net['blk_layers'],
                       in_chans=opt_net['in_channels'],
                       shallow_feats=opt_net['shallow_feats'],
                       pre_up_feats=opt_net['pre_up_feats'],
                       embed_dims=opt_net['embed_dims'],
                       patch_sizes=opt_net['patch_sizes'],
                       skip_dims=opt_net['skip_dims'],
                       drop_path_rate=opt_net['drop_path'] if mode == 'train' else 0.0,
                       use_checkpoint=opt_net['use_checkpoint'],
                       upsample_method=opt_net['upsample_method'])

        else:
            if opt['model_opt']['netG']['ct_size'] == 0:
                print("Using MTVNet without Carrier tokens!")
                from models.MTVNet_no_CT import MTVNet_no_CT as net
            else:
                from models.MTVNet_arch import MTVNet as net

            netG = net(input_size=input_size,
                      up_factor=opt['up_factor'],
                      num_levels=opt_net['num_levels'],
                      context_sizes=opt_net['context_sizes'],
                      num_blks=opt_net['num_blks'],
                      blk_layers=opt_net['blk_layers'],
                      in_chans=opt_net['in_channels'],
                      shallow_feats=opt_net['shallow_feats'],
                      pre_up_feats=opt_net['pre_up_feats'],
                      ct_embed_dims=opt_net['ct_embed_dims'],  # [512, 256, 128]
                      embed_dims=opt_net['embed_dims'],  # [512, 256, 128]
                      ct_size=opt_net['ct_size'],
                      ct_pool_method=opt_net['ct_pool_method'],
                      patch_sizes=opt_net['patch_sizes'],
                      num_heads=opt_net['num_heads'],
                      attn_window_sizes=opt_net['attn_window_sizes'],
                      enable_shift=opt_net['enable_shift'],
                      mlp_ratio=4.,
                      qkv_bias=True,
                      drop=0.,
                      attn_drop=0.,
                      drop_path=opt_net['drop_path'] if mode == 'train' else 0.0,
                      token_upsample_method=opt_net["token_upsample_method"],
                      upsample_method=opt_net["upsample_method"],
                      use_checkpoint=opt_net["use_checkpoint"],
                      layer_type=opt_net["layer_type"],  # fastervit_without_ct, swin, fastervit
                      enable_ape_ct=opt_net["enable_ape_ct"],
                      enable_ape_x=opt_net["enable_ape_x"],
                      enable_ct_rpb=opt_net["enable_ct_rpb"],
                      enable_conv_skip=opt_net["enable_conv_skip"],
                      patch_pe_method=opt_net["patch_pe_method"],)

    # ----------------------------------------
    # ArSSR
    # ----------------------------------------
    elif model_architecture == ("ArSSR"):
        from models.ArSSR import ArSSR as net
        netG = net(encoder_name=opt_net["encoder_name"],
                  feature_dim=opt_net["feature_dim"],
                  decoder_depth=opt_net["decoder_depth"],
                  decoder_width=opt_net["decoder_width"])

    # ----------------------------------------
    # ConvNextSR
    # ----------------------------------------
    elif model_architecture == ("ConvNeXtSR"):
        from models.ConvNeXtSR import ConvNeXtSR as net
        netG = net(up_factor=opt['up_factor'],
                   in_chans=opt_net['in_channels'],
                   depths=opt_net['depths'],
                   dims=opt_net['dims'],
                   growth_rate=opt_net['growth_rate'],
                   drop_path_rate=0.1 if mode == 'train' else 0.0,
                   layer_scale_init_value=1e-6,
                   upsample_method=opt_net['upsample_method'],
                   use_checkpoint=opt_net['use_checkpoint'])

    else:
        raise NotImplementedError('netG [{:s}] is not found.'.format(model_architecture))

    # ----------------------------------------
    # initialize weights
    # ----------------------------------------
    if opt['train_mode'] == 'scratch':
        init_weights(netG,
                     init_type=opt_net['init_type'],
                     init_bn_type=opt_net['init_bn_type'],
                     gain=opt_net['init_gain'])

    return netG


"""
# --------------------------------------------
# weights initialization
# --------------------------------------------
"""


def init_weights(net, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
    """
    # Kai Zhang, https://github.com/cszn/KAIR
    #
    # Args:
    #   init_type:
    #       default, none: pass init_weights
    #       normal; normal; xavier_normal; xavier_uniform;
    #       kaiming_normal; kaiming_uniform; orthogonal
    #   init_bn_type:
    #       uniform; constant
    #   gain:
    #       0.2
    """

    def init_fn(m, init_type='xavier_uniform', init_bn_type='uniform', gain=1):
        classname = m.__class__.__name__

        #if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # Here we search for layers called Conv3d, Conv2d or Linear.
        # We don't include ConvTranspose3d because we might want to use ICNR, which would otherwise be overwritten.
        if classname.find('Conv3d') != -1 or classname.find('Conv2d') != -1 or classname.find('Linear') != -1:

            if init_type == 'normal':
                init.normal_(m.weight.data, 0, 0.1)
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'uniform':
                init.uniform_(m.weight.data, -0.2, 0.2)
                m.weight.data.mul_(gain)

            elif init_type == 'xavier_normal':
                init.xavier_normal_(m.weight.data, gain=gain)
                m.weight.data.clamp_(-1, 1)

            elif init_type == 'xavier_uniform':
                init.xavier_uniform_(m.weight.data, gain=gain)

            elif init_type == 'kaiming_normal':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.clamp_(-1, 1).mul_(gain)

            elif init_type == 'kaiming_uniform':
                init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(gain)

            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)

            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))

            if m.bias is not None:
                m.bias.data.zero_()

        #elif classname.find('BatchNorm2d') != -1:
        elif classname.find('BatchNorm3d') != -1 or classname.find('BatchNorm2d') != -1:

            if init_bn_type == 'uniform':  # preferred
                if m.affine:
                    init.uniform_(m.weight.data, 0.1, 1.0)
                    init.constant_(m.bias.data, 0.0)
            elif init_bn_type == 'constant':
                if m.affine:
                    init.constant_(m.weight.data, 1.0)
                    init.constant_(m.bias.data, 0.0)
            else:
                raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_bn_type))

    if init_type not in ['default', 'none']:
        print('Initialization method [{:s} + {:s}], gain is [{:.2f}]'.format(init_type, init_bn_type, gain))
        fn = functools.partial(init_fn, init_type=init_type, init_bn_type=init_bn_type, gain=gain)
        net.apply(fn)
    else:
        # print('Pass this initialization! Initialization was done during network definition!')
        pass

