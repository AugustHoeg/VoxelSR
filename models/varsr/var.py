import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
import numpy as np
from torch.nn import functional as F
import dist
from models.basic_var import AdaLNBeforeHead
from models.basic_var import AdaLNSelfAttn_RoPE, precompute_freqs_cis, precompute_freqs_cis_cross, precompute_freqs_cis_zeros
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from models.diffusion.diffloss import DiffLoss
import scipy.stats as stats
import torch.utils.checkpoint as checkpoint
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class ControlNetConditioningEmbedding(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (128, 256, 512, 1024),
        return_rgbs: bool = False,
    ):
        super().__init__()

        self.return_rgbs = return_rgbs
        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        if return_rgbs:
            self.to_rgbs = nn.Conv2d(channel_out, 32, kernel_size=3, padding=1)

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)


    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding, inplace=True)

        out_rgbs = []
        for i, block in enumerate(self.blocks):
            embedding = block(embedding)
            embedding = F.silu(embedding, inplace=True)

        if self.return_rgbs:
            out_rgbs = self.to_rgbs(embedding)

        embedding = self.conv_out(embedding)

        return [embedding, out_rgbs] if self.return_rgbs else [embedding, None]


class VAR_RoPE(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, controlnet_depth=6, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[-1] ** 2 + 1
        self.begin_ends = []
        context_token = self.first_l
        self.context_token = context_token
        self.begin_ends.append((0, context_token))
        cur = context_token
        self.L = sum(pn**2 for pn in self.patch_nums)
        for i, pn in enumerate(self.patch_nums[1:]):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        self.last_level_pns = self.patch_nums[-1] ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        mask_ratio_min = 0.5
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.con_embedding = ControlNetConditioningEmbedding(self.C, 3, (32, 128, 256, 512, 1536) , return_rgbs=False)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        self.label_B_flag = 1
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)     
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        if self.label_B_flag:
            self.class_emb = nn.Embedding(self.num_classes, self.C)
            nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        else:   
            self.cond_proj = nn.Sequential(norm_layer(1024, elementwise_affine=False), nn.SiLU(inplace=False), nn.Linear(1024, self.C))
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)


        rope_patch_nums =  (self.patch_nums[-1], self.patch_nums[0], self.patch_nums[1]) +  self.patch_nums[2:]
        self.freqs_cis = precompute_freqs_cis(
            self.C // num_heads, rope_patch_nums
        )

        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.controlnet_depth = controlnet_depth
        self.interval = int(np.ceil(self.depth / self.controlnet_depth))
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn_RoPE(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            ) 
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat(
            [torch.full((context_token,), 0)]
            + [
                torch.full((pn * pn,), i + 1)
                for i, pn in enumerate(self.patch_nums[1:])
            ]
        ).view(1, self.L + context_token - 1, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(
            1, 1, self.L + context_token - 1, self.L + context_token - 1
        )
        self.register_buffer(
            "attn_bias_for_masking", attn_bias_for_masking.contiguous()
        )
        print(attn_bias_for_masking.shape)
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        self.decoder_norm = norm_layer(self.C)
        self.diffloss = DiffLoss(
            target_channels=self.Cvae,
            z_channels=self.C,
            width=1024,
            depth=6,
            num_sampling_steps='10',
            sampler='iddpm',
        )
        self.diffusion_batch_mul = 4
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def forward_diff_loss(self, z, target, mask=None):
        bs, seq_len, _ = target.shape
        target = target.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    def sample_orders(self, bsz):
        # generate a batch of random generation orders
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.last_level_pns)))
            #np.random.shuffle(order)
            orders.append(order)
        orders = torch.Tensor(np.array(orders)).cuda().long()
        return orders

    def random_masking(self, x, orders):
        # generate token mask
        bsz, seq_len, embed_dim = x.shape
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        # we cannot mask out all the tokens
        num_masked_tokens = 0#min(int(np.ceil(seq_len * mask_rate)), seq_len - 32)
        mask = torch.zeros(bsz, seq_len, device=x.device)
        # all first few stages are kept
        mask_keep = torch.zeros(
            bsz, self.L - seq_len + self.context_token - 1, device=x.device
        )
        mask = torch.scatter(
            mask,
            dim=-1,
            index=orders[:, :num_masked_tokens],
            src=torch.ones(bsz, seq_len, device=x.device),
        )
        mask_full = torch.cat([mask_keep, mask], dim=1).contiguous()
        return mask_full, mask

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, text_hidden, lr_inp, negative_text, label_B,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0, 
        more_smooth=False, lr_inp_scale=None, tile_flag=False,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng

        lr_inp, _ = self.con_embedding(lr_inp)
        lr_inp = lr_inp.view(B, self.C, -1).permute(0,2,1)
        assert lr_inp is not None
        assert lr_inp.shape[1] == self.context_token - 1

        sos = lr_inp.repeat(2, 1, 1)

        if self.label_B_flag:
            cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes-1)), dim=0))
        sos = torch.cat((sos, cond_BD.unsqueeze(1)), dim=1)

        lvl_pos = self.lvl_embed(self.lvl_1L)
        next_token_map = sos.expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        self.freqs_cis = self.freqs_cis.to(dist.get_device())
        
        cur_Lr = 1
        if lr_inp_scale is not None:
            next_token_map[:, -1, :] = next_token_map[:, -1, :] + self.word_embed(lr_inp_scale[:, 0]).repeat(2,1)
        for b in self.blocks: 
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            if si > 0:
                freqs_cis = self.freqs_cis[cur_L:cur_L + pn*pn]
                cur_L += pn * pn
            else:
                freqs_cis = self.freqs_cis[0:self.context_token]
                cur_L += self.context_token

            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            for i, b in enumerate(self.blocks):
                AdaLNSelfAttn_RoPE.forward
                x = b(x=x, cond_BD=cond_BD_or_gss,  freqs_cis=freqs_cis, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            if si == self.num_stages_minus_1:
                last_layer_cond = x
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                if lr_inp_scale is not None:
                    next_token_map = next_token_map + self.word_embed(lr_inp_scale[:, cur_Lr:cur_Lr + self.patch_nums[si+1] ** 2])
                    cur_Lr += self.patch_nums[si+1] ** 2
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: 
            b.attn.kv_caching(False)

        final_stage = 0
        if final_stage == 0:  
            last_stage_discrete_cond = self.vae_quant_proxy[0].embedding(idx_Bl)
            last_stage_discrete_cond = self.word_embed(last_stage_discrete_cond)
            last_stage_discrete_cond = torch.cat([last_stage_discrete_cond, last_stage_discrete_cond], dim=0)
            last_stage_cond = self.decoder_norm(last_layer_cond + last_stage_discrete_cond)
            bs, cur_seq_len, _ = last_stage_cond.shape
            last_stage_cond = last_stage_cond.reshape(bs * cur_seq_len, -1)
            h_BChw_diff = self.diffloss.sample(
                z=last_stage_cond, temperature=1.0, cfg=t
            )
            h_BChw_diff = h_BChw_diff.reshape(bs, cur_seq_len, -1)
            h_BChw_diff, _ = h_BChw_diff.chunk(2, dim=0)
            h_BChw_diff = h_BChw_diff.reshape(B, 1024, -1).transpose(1, 2).reshape(
                B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]
            )
            f_hat += h_BChw_diff

        if tile_flag:
            return f_hat
        else:
            return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

    def forward(self, x_BLCv_wo_first_l: torch.Tensor, label_B, lr_inp, text_hidden,
        last_layer_gt: torch.Tensor = None,
        last_layer_gt_discrete: torch.Tensor = None,
        lr_inp_scale = None,
    ) -> torch.Tensor:  # returns logits_BLV
        bg, ed = (
            self.begin_ends[self.prog_si]
            if self.prog_si >= 0
            else (0, self.L + self.context_token - 1)
        )
        B = x_BLCv_wo_first_l.shape[0]
        orders = self.sample_orders(bsz=B)
        mask, mask_wo_prev_stages = self.random_masking(
            x_BLCv_wo_first_l[:, -self.last_level_pns :, :], orders
        )
        mask = (1 - mask).nonzero(as_tuple=True)
        mask_wo_prev_stages = (1 - mask_wo_prev_stages).nonzero(as_tuple=True)
        last_layer_gt = last_layer_gt[mask_wo_prev_stages].reshape(B, -1, last_layer_gt.shape[-1])
        last_layer_gt_discrete = last_layer_gt_discrete[mask_wo_prev_stages].reshape(B, -1)
        ed = (
            last_layer_gt.shape[1]
            + self.L
            + self.context_token
            - 1
            - self.last_level_pns
        )

        with torch.cuda.amp.autocast(enabled=False):
            lr_inp, out_rgbs = self.con_embedding(lr_inp)
            sos = lr_inp.view(B, self.C, -1).permute(0,2,1)
            if self.label_B_flag:
                cond_BD = self.class_emb(label_B)
            sos = torch.cat((sos, cond_BD.unsqueeze(1)), dim=1)
            sos = sos.expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else:
                x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
                if lr_inp_scale is not None:
                    x_BLC[:, -self.L:, :] = x_BLC[:, -self.L:, :] + self.word_embed(lr_inp_scale.float())
            x_BLC = x_BLC[mask].reshape(B, -1, x_BLC.shape[-1])
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))
    
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        self.freqs_cis = self.freqs_cis.to(x_BLC.device)
        freqs_cis = self.freqs_cis.repeat(B, 1, 1)[mask].view(B, ed, -1)
        
        AdaLNSelfAttn_RoPE.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss,  
                    freqs_cis=freqs_cis, attn_bias=attn_bias)
        last_layer_cond = x_BLC[:, self.L + self.context_token - 1 - self.last_level_pns :, :]
        x_BLC_logits = self.get_logits(x_BLC.float(), cond_BD)
        x_BLC = x_BLC_logits[:, self.context_token - 1 :, :]
        
        with torch.no_grad():
            try:
                idx_BL_sampled = sample_with_top_k_top_p_(
                    x_BLC_logits[
                        :, self.L + self.context_token - 1 - self.last_level_pns :
                    ]
                    .clone()
                    .detach(),
                    rng=self.rng,
                    top_k=1,
                    top_p=0.96,
                    num_samples=1,
                )[:, :, 0]
            except:
                idx_BL_sampled = last_layer_gt_discrete
        
        last_stage_discrete_embed = self.vae_quant_proxy[0].embedding(idx_BL_sampled)
        last_stage_discrete_cond = self.word_embed(last_stage_discrete_embed)
        last_layer_cond = self.decoder_norm(last_layer_cond + last_stage_discrete_cond)

        last_layer_gt_continuous = last_layer_gt - last_stage_discrete_embed
        diff_loss = self.forward_diff_loss(
            z=last_layer_cond, target=last_layer_gt_continuous
        )
        return (x_BLC, diff_loss, out_rgbs, mask_wo_prev_stages)    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            if hasattr(self.head_nm, 'ada_lin'):
                self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
                if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                    self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            if isinstance(sab, AdaLNSelfAttn_RoPE):
                sab: AdaLNSelfAttn_RoPE
                sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
                sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
                if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                    nn.init.ones_(sab.ffn.fcg.bias)
                    nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
                if hasattr(sab, 'ada_lin'):
                    sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                    sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                    if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                        sab.ada_lin[-1].bias.data.zero_()
                elif hasattr(sab, 'ada_gss'):
                    sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                    sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
            elif isinstance(sab, BaseBlock_RoPE):
                if sab.paca_flag:
                    for sab in [sab.self_attn]:
                        sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
                        sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
                        if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                            nn.init.ones_(sab.ffn.fcg.bias)
                            nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
                        if hasattr(sab, 'ada_lin'):
                            sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                            sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                            if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                                sab.ada_lin[-1].bias.data.zero_()
                        elif hasattr(sab, 'ada_gss'):
                            sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                            sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'



class ImgVAR_RoPE(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, controlnet_depth=6, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        context_token = self.first_l
        self.context_token = context_token
        self.begin_ends.append((0, context_token))
        cur = context_token
        self.L = sum(pn**2 for pn in self.patch_nums)
        for i, pn in enumerate(self.patch_nums[1:]):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        self.last_level_pns = self.patch_nums[-1] ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        self.label_B_flag = 1
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)     
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        if self.label_B_flag:
            self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
            nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        else:   
            self.cond_proj = nn.Sequential(norm_layer(1024, elementwise_affine=False), nn.SiLU(inplace=False), nn.Linear(1024, self.C))
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)


        # 3. absolute position embedding
        rope_patch_nums =  (self.patch_nums[0], self.patch_nums[1]) +  self.patch_nums[2:]
        self.freqs_cis = precompute_freqs_cis(
            self.C // num_heads, rope_patch_nums
        )
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.controlnet_depth = controlnet_depth
        self.interval = int(np.ceil(self.depth / self.controlnet_depth))
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn_RoPE(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            ) 
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat(
            [torch.full((context_token,), 0)]
            + [
                torch.full((pn * pn,), i + 1)
                for i, pn in enumerate(self.patch_nums[1:])
            ]
        ).view(1, self.L + context_token - 1, 1)
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(
            1, 1, self.L + context_token - 1, self.L + context_token - 1
        )
        self.register_buffer(
            "attn_bias_for_masking", attn_bias_for_masking.contiguous()
        )
        print(attn_bias_for_masking.shape)
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

        self.decoder_norm = norm_layer(self.C)
        self.diffloss = DiffLoss(
            target_channels=self.Cvae,
            z_channels=self.C,
            width=1024,
            depth=6,
            num_sampling_steps='10',
            sampler='iddpm',
        )
        self.diffusion_batch_mul = 4
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def forward_diff_loss(self, z, target, mask=None):
        bs, seq_len, _ = target.shape
        target = target.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bs * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        loss = self.diffloss(z=z, target=target, mask=mask)
        return loss

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, text_hidden, negative_text, label_B,
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0, 
        more_smooth=False, gt_BL=None,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)

        if self.label_B_flag:
            sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L)
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]
        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        self.freqs_cis = self.freqs_cis.to(dist.get_device())
        
        for b in self.blocks: 
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = (si+1) / self.num_stages_minus_1
            if si > 0:
                freqs_cis = self.freqs_cis[cur_L:cur_L + pn*pn]
                cur_L += pn * pn
            else:
                freqs_cis = self.freqs_cis[0:self.context_token]
                cur_L += self.context_token

            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            for i, b in enumerate(self.blocks):
                AdaLNSelfAttn_RoPE.forward
                x = b(x=x, cond_BD=cond_BD_or_gss,  freqs_cis=freqs_cis, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
            if si == self.num_stages_minus_1:
                last_layer_cond = x
            
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

            if si == 0:
                logits_BlV = logits_BlV[:, [-1], :]
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: 
            b.attn.kv_caching(False)

        final_stage = 0
        if final_stage == 0:  
            last_stage_discrete_cond = self.vae_quant_proxy[0].embedding(idx_Bl)
            last_stage_discrete_cond = self.word_embed(last_stage_discrete_cond)
            last_stage_discrete_cond = torch.cat([last_stage_discrete_cond, last_stage_discrete_cond], dim=0)
            last_stage_cond = self.decoder_norm(last_layer_cond + last_stage_discrete_cond)
            bs, cur_seq_len, _ = last_stage_cond.shape
            last_stage_cond = last_stage_cond.reshape(bs * cur_seq_len, -1)
            h_BChw_diff = self.diffloss.sample(
                z=last_stage_cond, temperature=1.0, cfg=t
            )
            h_BChw_diff = h_BChw_diff.reshape(bs, cur_seq_len, -1)
            h_BChw_diff, _ = h_BChw_diff.chunk(2, dim=0)
            h_BChw_diff = h_BChw_diff.reshape(B, 1024, -1).transpose(1, 2).reshape(
                B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]
            )
            f_hat += h_BChw_diff
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]

    def forward(self, x_BLCv_wo_first_l: torch.Tensor, label_B, text_hidden,
        last_layer_gt: torch.Tensor = None,
        last_layer_gt_discrete: torch.Tensor = None,
    ) -> torch.Tensor:  # returns logits_BLV
        bg, ed = (
            self.begin_ends[self.prog_si]
            if self.prog_si >= 0
            else (0, self.L + self.context_token - 1)
        )
        B = x_BLCv_wo_first_l.shape[0]

        with torch.cuda.amp.autocast(enabled=False):
            if self.label_B_flag:
                label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
                sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1))# lvl: BLC;  pos: 1LC
        
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        self.freqs_cis = self.freqs_cis.to(x_BLC.device)
        freqs_cis = self.freqs_cis[:ed]
        
        AdaLNSelfAttn_RoPE.forward
        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss,  freqs_cis=freqs_cis, attn_bias=attn_bias)
        last_layer_cond = x_BLC[:, self.L + self.context_token - 1 - self.last_level_pns :, :]
        x_BLC_logits = self.get_logits(x_BLC.float(), cond_BD)
        x_BLC = x_BLC_logits[:, self.context_token - 1 :, :]
        
        with torch.no_grad():
            try:
                idx_BL_sampled = sample_with_top_k_top_p_(
                    x_BLC_logits[
                        :, self.L + self.context_token - 1 - self.last_level_pns :
                    ]
                    .clone()
                    .detach(),
                    rng=self.rng,
                    top_k=1,
                    top_p=0.96,
                    num_samples=1,
                )[:, :, 0]
            except:
                idx_BL_sampled = last_layer_gt_discrete
        
        last_stage_discrete_embed = self.vae_quant_proxy[0].embedding(idx_BL_sampled)
        last_stage_discrete_cond = self.word_embed(last_stage_discrete_embed)
        last_layer_cond = self.decoder_norm(last_layer_cond + last_stage_discrete_cond)

        last_layer_gt_continuous = last_layer_gt - last_stage_discrete_embed
        diff_loss = self.forward_diff_loss(
            z=last_layer_cond, target=last_layer_gt_continuous
        )
        return (x_BLC, diff_loss)    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            if hasattr(self.head_nm, 'ada_lin'):
                self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
                if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                    self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            if isinstance(sab, AdaLNSelfAttn_RoPE):
                sab: AdaLNSelfAttn_RoPE
                sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
                sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
                if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                    nn.init.ones_(sab.ffn.fcg.bias)
                    nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
                if hasattr(sab, 'ada_lin'):
                    sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                    sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                    if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                        sab.ada_lin[-1].bias.data.zero_()
                elif hasattr(sab, 'ada_gss'):
                    sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                    sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
            elif isinstance(sab, BaseBlock_RoPE):
                if sab.paca_flag:
                    for sab in [sab.self_attn]:
                        sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
                        sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
                        if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                            nn.init.ones_(sab.ffn.fcg.bias)
                            nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
                        if hasattr(sab, 'ada_lin'):
                            sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                            sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                            if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                                sab.ada_lin[-1].bias.data.zero_()
                        elif hasattr(sab, 'ada_gss'):
                            sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                            sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'
