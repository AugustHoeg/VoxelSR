"""Core OSEDiff networks, vendored from https://github.com/cswry/OSEDiff.

This is a faithful port of the generator / VSD-regularizer / test networks from
upstream ``osediff.py``. The one-step SR algorithm is unchanged:

    LR image  --VAE.encode-->  latent  --UNet (t=999, single step)-->  eps
           --scheduler.step-->  x0 latent  --VAE.decode-->  SR image

LoRA adapters are injected into the (otherwise frozen) VAE encoder and the UNet
exactly as upstream (:func:`initialize_vae` / :func:`initialize_unet`), and the
VAE *decoder* stays frozen. Variational Score Distillation (VSD) is provided by
:class:`OSEDiff_reg` (a trainable "fake-score" UNet + a frozen pretrained UNet),
identical to upstream's ``OSEDiff_reg``.

Deliberate, documented differences from upstream (none touch the algorithm):

* ``AutoencoderKL`` / ``UNet2DConditionModel`` are imported from **stock
  diffusers** instead of upstream's vendored copies. Those copies were byte-for-
  byte stock diffusers with three extra LoRA ``merge_and_unload`` helpers
  appended; modern diffusers + peft provide merging, so carrying ~3k lines of
  frozen diffusers source would only rot. Test-time LoRA merging is handled via
  peft in :class:`OSEDiff_test` when ``merge_and_unload_lora`` is set.
* Device is taken from ``args.device`` (default ``"cuda"``) rather than the
  hard-coded ``.cuda()`` / ``"cuda"`` literals upstream, so the module is
  device-agnostic (CPU smoke tests, DDP device placement).
* :class:`OSEDiff_reg` takes explicit ``device`` / ``weight_dtype`` instead of a
  HuggingFace ``accelerator`` object (VoxelSR drives training via its own
  ModelBase / DDP, not ``accelerate``). The logic is otherwise unchanged.
* Tiled-VAE inference (upstream ``my_utils.vaehook``) is an inference-memory
  optimization for large images and is **optional** here (``args.use_tiled_vae``,
  default off); VoxelSR operates on small slice patches that never need tiling.
* Upstream's ``OSEDiff_inference_time`` class (a non-tiled clone of
  ``OSEDiff_test`` for latency benchmarking) is dropped as non-essential.

Grayscale->RGB channel handling and null-prompt conditioning are *not* done here
(this module stays faithful to the 3-channel, prompt-driven upstream); they live
in the VoxelSR ModelBase wrapper ``models/model_osediff.py``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from peft import LoraConfig


def initialize_vae(args):
    """Freeze the VAE and add a LoRA adapter to its encoder (+ quant_conv).

    Faithful to upstream: the decoder and ``post_quant_conv`` are left without
    adapters so the VAE decoder stays frozen.
    """
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae.requires_grad_(False)
    vae.train()

    l_target_modules_encoder = []
    l_grep = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out", "to_k", "to_q", "to_v", "to_out.0"]
    for n, p in vae.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("encoder" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
            elif ('quant_conv' in n) and ('post_quant_conv' not in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder)
    vae.add_adapter(lora_conf_encoder, adapter_name="default_encoder")

    return vae, l_target_modules_encoder


def initialize_unet(args, return_lora_module_names=False, pretrained_model_name_or_path=None):
    """Freeze the UNet and add three LoRA adapters (encoder / decoder / others)."""
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n:
            continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight", ""))
                break
            elif pattern in n and ("up_blocks" in n or "conv_out" in n):
                l_target_modules_decoder.append(n.replace(".weight", ""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight", ""))
                break

    lora_conf_encoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=args.lora_rank, init_lora_weights="gaussian", target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")

    return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others


class OSEDiff_gen(torch.nn.Module):
    """One-step generator: LoRA VAE-encoder + LoRA UNet + frozen VAE-decoder."""

    def __init__(self, args):
        super().__init__()

        self.device = getattr(args, "device", "cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").to(self.device)
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device=self.device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self.args = args

        self.vae, self.lora_vae_modules_encoder = initialize_vae(self.args)
        self.unet, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = initialize_unet(self.args)
        self.lora_rank_unet = self.args.lora_rank
        self.lora_rank_vae = self.args.lora_rank

        self.unet.to(self.device)
        self.vae.to(self.device)
        self.timesteps = torch.tensor([999], device=self.device).long()
        # modern-diffusers compat: pin the single inference timestep to 999 so
        # DDPMScheduler.step(t=999) resolves prev_t=-1 (pure x0). Upstream ran on
        # diffusers 0.25 where step() computed prev_t = 999 - 1000 = -1 directly;
        # diffusers >=0.30 instead looks timestep up in scheduler.timesteps.
        self.noise_scheduler.timesteps = self.timesteps
        self.text_encoder.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        self.vae.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n:
                _p.requires_grad = True
        self.unet.conv_in.requires_grad_(True)
        for n, _p in self.vae.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    def forward(self, c_t, batch=None, args=None):

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor

        # calculate prompt_embeddings and neg_prompt_embeddings
        prompt_embeds = self.encode_prompt(batch["prompt"])
        neg_prompt_embeds = self.encode_prompt(batch["neg_prompt"])

        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=prompt_embeds.to(torch.float32),).sample
        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, x_denoised, prompt_embeds, neg_prompt_embeds

    def save_model(self, outf):
        sd = {}
        sd["vae_lora_encoder_modules"] = self.lora_vae_modules_encoder
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] = \
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["rank_vae"] = self.lora_rank_vae
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or "conv_in" in k}
        sd["state_dict_vae"] = {k: v for k, v in self.vae.state_dict().items() if "lora" in k}
        torch.save(sd, outf)


class OSEDiff_reg(torch.nn.Module):
    """VSD regularizer: trainable "fake-score" UNet + frozen pretrained UNet.

    Upstream took a HuggingFace ``accelerator`` to derive device / mixed-precision
    dtype; here those are passed explicitly so the module is decoupled from
    ``accelerate`` (VoxelSR trains via its own ModelBase / DDP). The VSD loss math
    (:meth:`diff_loss`, :meth:`distribution_matching_loss`) is unchanged.
    """

    def __init__(self, args, device="cuda", weight_dtype=torch.float32):
        super().__init__()

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.args = args

        self.weight_dtype = weight_dtype

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = \
            initialize_unet(args)

        self.text_encoder.to(self.device, dtype=weight_dtype)
        self.unet_fix.to(self.device, dtype=weight_dtype)
        self.unet_update.to(self.device)
        self.vae.to(self.device)

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def diff_loss(self, latents, prompt_embeds, args):

        latents, prompt_embeds = latents.detach(), prompt_embeds.detach()
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        noise_pred = self.unet_update(
            noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
        ).sample

        loss_d = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):

        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def distribution_matching_loss(self, latents, prompt_embeds, neg_prompt_embeds, args):
        bsz = latents.shape[0]
        timesteps = torch.randint(20, 980, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        with torch.no_grad():

            noise_pred_update = self.unet_update(
                noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds.float(),
            ).sample

            x0_pred_update = self.eps_to_mu(self.noise_scheduler, noise_pred_update, noisy_latents, timesteps)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timesteps_input = torch.cat([timesteps] * 2)
            prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)

            noise_pred_fix = self.unet_fix(
                noisy_latents_input.to(dtype=self.weight_dtype),
                timestep=timesteps_input,
                encoder_hidden_states=prompt_embeds.to(dtype=self.weight_dtype),
            ).sample

            noise_pred_uncond, noise_pred_text = noise_pred_fix.chunk(2)
            noise_pred_fix = noise_pred_uncond + args.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred_fix.to(dtype=torch.float32)

            x0_pred_fix = self.eps_to_mu(self.noise_scheduler, noise_pred_fix, noisy_latents, timesteps)

        weighting_factor = torch.abs(latents - x0_pred_fix).mean(dim=[1, 2, 3], keepdim=True)

        grad = (x0_pred_update - x0_pred_fix) / weighting_factor
        loss = F.mse_loss(latents, (latents - grad).detach())

        return loss


class OSEDiff_test(torch.nn.Module):
    """Inference network: loads a trained OSEDiff checkpoint and runs one step.

    Tiled-VAE (upstream ``my_utils.vaehook``) is gated behind ``args.use_tiled_vae``
    (default off); with it off the forward is the plain non-tiled one-step path,
    which is all that small slice patches need.
    """

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.device = getattr(args, "device", 'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="text_encoder")
        self.noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
        self.noise_scheduler.set_timesteps(1, device=self.device)
        self.vae = AutoencoderKL.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(self.args.pretrained_model_name_or_path, subfolder="unet")

        # optional tiled VAE (large-image inference only)
        if getattr(args, "use_tiled_vae", False):
            self._init_tiled_vae(encoder_tile_size=args.vae_encoder_tiled_size, decoder_tile_size=args.vae_decoder_tiled_size)

        self.weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            self.weight_dtype = torch.float16

        osediff = torch.load(args.osediff_path)
        self.load_ckpt(osediff)

        # merge lora
        if self.args.merge_and_unload_lora:
            print('===> MERGE LORA <===')
            self.vae = self.vae.merge_and_unload()
            self.unet = self.unet.merge_and_unload()

        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.timesteps = torch.tensor([999], device=self.device).long()
        # modern-diffusers compat (see OSEDiff_gen.__init__): pin the single
        # inference timestep so DDPMScheduler.step(t=999) resolves prev_t=-1.
        self.noise_scheduler.timesteps = self.timesteps
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

    def load_ckpt(self, model):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=model["rank_unet"], init_lora_weights="gaussian", target_modules=model["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or "conv_in" in n:
                p.data.copy_(model["state_dict_unet"][n])
        self.unet.set_adapter(["default_encoder", "default_decoder", "default_others"])

        # load vae lora
        vae_lora_conf_encoder = LoraConfig(r=model["rank_vae"], init_lora_weights="gaussian", target_modules=model["vae_lora_encoder_modules"])
        self.vae.add_adapter(vae_lora_conf_encoder, adapter_name="default_encoder")
        for n, p in self.vae.named_parameters():
            if "lora" in n:
                p.data.copy_(model["state_dict_vae"][n])
        self.vae.set_adapter(['default_encoder'])

    def encode_prompt(self, prompt_batch):
        prompt_embeds_list = []
        with torch.no_grad():
            for caption in prompt_batch:
                text_input_ids = self.tokenizer(
                    caption, max_length=self.tokenizer.model_max_length,
                    padding="max_length", truncation=True, return_tensors="pt"
                ).input_ids
                prompt_embeds = self.text_encoder(
                    text_input_ids.to(self.text_encoder.device),
                )[0]
                prompt_embeds_list.append(prompt_embeds)
        prompt_embeds = torch.concat(prompt_embeds_list, dim=0)
        return prompt_embeds

    @torch.no_grad()
    def forward(self, lq, prompt):

        prompt_embeds = self.encode_prompt([prompt])
        lq_latent = self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample() * self.vae.config.scaling_factor
        ## add tile function
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.latent_tiled_size, self.args.latent_tiled_overlap)
        if h * w <= tile_size * tile_size:
            print("[Tiled Latent]: the input size is tiny and unnecessary to tile.")
            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds).sample
        else:
            print(f"[Tiled Latent]: the input size is {lq.shape[-2]}x{lq.shape[-1]}, need to tiled")
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size - tile_overlap * grid_rows, 0) + tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size - tile_overlap * grid_cols, 0) + tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols - 1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        model_out = self.unet(input_list_t, self.timesteps, encoder_hidden_states=prompt_embeds.to(self.weight_dtype),).sample
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols - 1 or row < grid_rows - 1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size - tile_overlap * row, 0)
                        ofs_y = max(col * tile_size - tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows - 1:
                        ofs_x = w - tile_size
                    if col == grid_cols - 1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row * grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.noise_scheduler.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        output_image = (self.vae.decode(x_denoised.to(self.weight_dtype) / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image

    def _init_tiled_vae(self,
                        encoder_tile_size=256,
                        decoder_tile_size=256,
                        fast_decoder=False,
                        fast_encoder=False,
                        color_fix=False,
                        vae_to_gpu=True):
        from models.osediff.vaehook import VAEHook
        # save original forward (only once)
        if not hasattr(self.vae.encoder, 'original_forward'):
            setattr(self.vae.encoder, 'original_forward', self.vae.encoder.forward)
        if not hasattr(self.vae.decoder, 'original_forward'):
            setattr(self.vae.decoder, 'original_forward', self.vae.decoder.forward)

        encoder = self.vae.encoder
        decoder = self.vae.decoder

        self.vae.encoder.forward = VAEHook(
            encoder, encoder_tile_size, is_decoder=False, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)
        self.vae.decoder.forward = VAEHook(
            decoder, decoder_tile_size, is_decoder=True, fast_decoder=fast_decoder, fast_encoder=fast_encoder, color_fix=color_fix, to_gpu=vae_to_gpu)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)) / sqrt(2 * pi * var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y - midpoint) * (y - midpoint) / (latent_height * latent_height) / (2 * var)) / sqrt(2 * pi * var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))
