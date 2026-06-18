

"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt, mode, data_parallel=True):
    model = opt['model_opt']['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M

    elif model == 'gan':     # one input: L
        from models.model_gan import ModelGAN as M

    elif model == 'ragan':
        from models.model_ragan import ModelRaGAN as M

    elif model == 'vqvae':     # one input: L
        from models.model_vqvae import ModelVQVAE as M

    elif model == 'vqgan':     # one input: L
        from models.model_vqgan import ModelVQGAN as M

    elif model == 'transformer_vq':
        from models.model_vq_transformer import ModelTransformerVQ as M

    elif model == 'mask_transformer':
        from models.model_mask_transformer import ModelMaskTransformer as M

    elif model == 'maskvsrt':
        from models.model_maskvsrt import ModelMaskVSRT as M

    elif model == 'aesop':
        from models.model_aesop import ModelAESOP as M

    elif model == "degradation":
        from models.model_degradation import ModelDegradation as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt, mode, data_parallel=data_parallel)

    if mode == 'train' and opt['rank'] == 0:
        print('Training model [{:s}] is created.'.format(m.__class__.__name__))

    return m