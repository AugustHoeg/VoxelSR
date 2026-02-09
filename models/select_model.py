

"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt, mode):
    model = opt['model_opt']['model']      # one input: L

    if model == 'plain':
        from models.model_plain import ModelPlain as M

    elif model == 'gan' or model == 'ragan':     # one input: L
        from models.model_gan import ModelGAN as M

    elif model == 'wgan-gp':     # one input: L
        from models.model_wgan_gp import ModelWGAN_GP as M

    elif model == "implicit":
        from models.model_implicit import ModelImplicit as M

    elif model == "degradation":
        from models.model_degradation import ModelDegradation as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt, mode)

    if mode == 'train':
        print('Training model [{:s}] is created.'.format(m.__class__.__name__))

    return m