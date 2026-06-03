import torch.nn.functional as F
import torch.nn as nn
import torch
import types 
from einops import rearrange, repeat
import os

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
# sys.path.append("../src")
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config


from diffusers import CogVideoXTransformer3DModel

# from diffusers.models.transformers.cogvideox_transformer_3d_wxd import CogVideoXTransformer3DModel

class Discriminator(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, dim=1920, h=30, w=45):
        super().__init__()
        self.h = h
        self.w = w
        self.cls_pred_branch = nn.Sequential(
            nn.Conv2d(kernel_size=(4,6), in_channels=dim, out_channels=dim, stride=(2,3), padding=(4,6)), # 30x45 -> 16x16
            nn.GroupNorm(num_groups=32, num_channels=dim),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=dim, out_channels=dim, stride=2, padding=1), # 16x16 -> 8x8 
            nn.GroupNorm(num_groups=32, num_channels=dim),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=dim, out_channels=dim, stride=2, padding=1), # 8x8 -> 4x4
            nn.GroupNorm(num_groups=32, num_channels=dim),
            nn.SiLU(),
            nn.Conv2d(kernel_size=4, in_channels=dim, out_channels=dim, stride=4, padding=0), # 4x4 -> 1x1
            nn.GroupNorm(num_groups=32, num_channels=dim),
            nn.SiLU(),
            nn.Conv2d(kernel_size=1, in_channels=dim, out_channels=1, stride=1, padding=0), # 1x1 -> 1x1
        )
    def forward(self, rep):
        rep = rearrange(rep, "b (f h w) c -> (b f) c h w", h=self.h, w=self.w)
        logits = self.cls_pred_branch(rep).squeeze(dim=[2, 3])
        return logits

class UnifiedModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        load_dtype = torch.bfloat16 if "5b" in args.pretrained_model_name_or_path.lower() else torch.float16
        self.feedforward_model = CogVideoXTransformer3DModel.from_pretrained(
            # "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-1219-fbf-noaug/checkpoint-6000", # origin coarse
            # "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-0109-fbf-fft/checkpoint-1200", # fft coarse model
            "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-0110-fbf-fft-inputfft-resume-0109/checkpoint-1400", # input-fft-fft-loss coarse model
            subfolder="transformer",
            torch_dtype=load_dtype,
            revision=args.revision,
            variant=args.variant,
            in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        self.fake_unet = CogVideoXTransformer3DModel.from_pretrained(
            # "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-1219-fbf-noaug/checkpoint-6000",
            "/data/wangxd/ckpt/cogvideox-A4-clean-image-inject-f13-fps1-0109-fbf-fft/checkpoint-1200",
            subfolder="transformer",
            # "/data/wuzhirong/ckpts/cogvideox-A4-clean-image-distill-gan-0101-new-5-onlyguidance/checkpoint-1190/fake_unet",
            torch_dtype=load_dtype,
            revision=args.revision,
            variant=args.variant,
            in_channels=32, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
        )
        from classify_forward import classify_forward
        self.fake_unet.forward = types.MethodType(
            classify_forward, self.fake_unet
        )
        self.discriminator = Discriminator()
        # self.discriminator = Discriminator.from_pretrained("/data/wuzhirong/ckpts/cogvideox-A4-clean-image-distill-gan-0101-new-5-onlyguidance/checkpoint-1190/discriminator")

    def frozen_guidance_model(self):
        self.fake_unet.requires_grad_(False)
        self.discriminator.requires_grad_(False)
    def unfrozen_guidance_model(self):
        self.fake_unet.requires_grad_(True)
        self.discriminator.requires_grad_(True)

    def save_model(self, output_dir):
        os.makedirs(output_dir,exist_ok=True)
        self.feedforward_model.save_pretrained(os.path.join(output_dir, "feedforward_model"))
        self.fake_unet.save_pretrained(os.path.join(output_dir, "fake_unet"))
        self.discriminator.save_pretrained(os.path.join(output_dir, "discriminator"))
    
    def enable_gradient_checkpointing(self):
        self.feedforward_model.enable_gradient_checkpointing()
        self.fake_unet.enable_gradient_checkpointing()
        
    def compute_cls_logits(self, 
                           latent,         
                           coarse_image_latents,
                           image_latent,
                           prompt_embeds,
                           image_rotary_emb,

                           scheduler
                           ):
        batch_size = latent.shape[0]
        weight_dtype = coarse_image_latents.dtype

        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (batch_size,), device=coarse_image_latents.device
        )
        timesteps = timesteps.long()
        
        noise = torch.randn_like(latent).to(weight_dtype)
        noisy_coarse_video_latents = scheduler.add_noise(latent, noise, timesteps)
        noisy_model_input = torch.cat([noisy_coarse_video_latents, coarse_image_latents], dim=2)
        rep = self.fake_unet(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
            image_latent = image_latent,
            classify_mode=True
        )[0]
        return self.discriminator(rep)

    def compute_distribution_matching_loss(
        self, 
        clean_x_0,
        continue_video_latents,
        coarse_image_latents,
        image_latent,
        prompt_embeds,
        image_rotary_emb,

        real_unet,
        scheduler,
    ):
        batch_size = 1
        weight_dtype = continue_video_latents.dtype

        origin_clean_x_0 = clean_x_0
        
        with torch.no_grad():
            # get timestep, then DMD loss
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (batch_size,), device=continue_video_latents.device
            )
            timesteps = timesteps.long()
            
            noise = torch.randn_like(clean_x_0).to(weight_dtype)

            # student pred x^hat_0
            noisy_coarse_video_latents = scheduler.add_noise(clean_x_0, noise, timesteps)
            noisy_model_input = torch.cat([noisy_coarse_video_latents, coarse_image_latents], dim=2)
            model_output = self.fake_unet(
                hidden_states=noisy_model_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                image_rotary_emb=image_rotary_emb,
                return_dict=False,
                image_latent = image_latent,
            )[0]
            fake_output = scheduler.get_velocity(model_output, noisy_coarse_video_latents, timesteps) # student clean x_0
            
            # import pdb; pdb.set_trace()
            # teacher pred x^hat_0
            # lack continue pred, here use GT internal continue latent, [pred_x0, GT_latent, pred_x0]
            # continue_video_latents # [B*L, 13, C, H, W]
            # insert clean_x_0 # b (l f) c h w
            post_continue_video_latents = []
            for b_idx in range(continue_video_latents.shape[0]):
                post_continue_video_latents.append(torch.cat([ clean_x_0[0, b_idx: b_idx+1], continue_video_latents[b_idx, 1:-1], clean_x_0[0, b_idx+1: b_idx+2]]) ) # [13, c, h, w]
            
            post_continue_video_latents = torch.stack(post_continue_video_latents, dim=0)
            
            # take previous 12 continue_video_latents
            clean_teacher = []
            for b_idx in range(post_continue_video_latents.shape[0]):
                current_continue_latents = post_continue_video_latents[b_idx:b_idx+1]
                noisy_current_continue_latents = scheduler.add_noise(current_continue_latents, noise, timesteps)

                # noisy_model_input = torch.cat([noisy_current_continue_latents, continue_video_latents[b_idx, :1]], dim=2)
                current_continue_image_latent = continue_video_latents[b_idx:b_idx+1, :1]

                padding_shape = (current_continue_latents.shape[0], current_continue_latents.shape[1] - 1, *current_continue_latents.shape[2:])
                latent_padding = current_continue_image_latent.new_zeros(padding_shape)
                current_continue_image_latent = torch.cat([current_continue_image_latent, latent_padding], dim=1)
                
                noisy_model_input = torch.cat([noisy_current_continue_latents, current_continue_image_latent], dim=2)
                noisy_model_input = noisy_model_input.to(weight_dtype)

                current_output = real_unet(
                    hidden_states=noisy_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timesteps,
                    image_rotary_emb=image_rotary_emb,
                    return_dict=False,
                    image_latent=current_continue_image_latent[:, :1],
                )[0]

                # [1, 13, c, h, w]
                current_clean_teach = scheduler.get_velocity(current_output, noisy_current_continue_latents, timesteps)

                if b_idx == 0:
                    clean_teacher.append(current_clean_teach[:, :1])
                    clean_teacher.append(current_clean_teach[:, -1:])
                else:
                    clean_teacher.append(current_clean_teach[:, -1:])

            real_output = torch.cat(clean_teacher, dim=1) # [b, 13, c, h, w]
            

            p_fake = (clean_x_0 - fake_output) # [b, 13, C, H, W]
            p_real = (clean_x_0 - real_output) # [b, 13, C, H, W]

            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True) 
            grad = torch.nan_to_num(grad)
        
        distill_loss = 0.5 * F.mse_loss(clean_x_0.float(), (origin_clean_x_0-grad).detach().float(), reduction="mean") 

        return distill_loss

    def compute_generator_clean_cls_loss(self, 
        fake_image, 
        coarse_image_latents,
        image_latent,
        prompt_embeds,
        image_rotary_emb,

        scheduler,
    ):
        pred_realism_on_fake_with_grad = self.compute_cls_logits(
            fake_image, 
            coarse_image_latents,
            image_latent,
            prompt_embeds,
            image_rotary_emb,

            scheduler,
        )
        gen_cls_loss = F.softplus(-pred_realism_on_fake_with_grad).mean()
        return gen_cls_loss

    def compute_loss_fake(
        self,
        fake_image,
        real_image,
        coarse_image_latents,
        image_latent,
        prompt_embeds,
        image_rotary_emb,

        scheduler,
    ):
        # self.fake_unet.enable_gradient_checkpointing()

        fake_image = fake_image.detach()
        batch_size = fake_image.shape[0]
        weight_dtype = coarse_image_latents.dtype

        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (batch_size,), device=coarse_image_latents.device
        )
        timesteps = timesteps.long()
        
        noise = torch.randn_like(fake_image).to(weight_dtype)
        noisy_coarse_video_latents = scheduler.add_noise(fake_image, noise, timesteps)
        noisy_model_input = torch.cat([noisy_coarse_video_latents, coarse_image_latents], dim=2)

        model_output = self.fake_unet(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
            image_latent = image_latent,
        )[0]
        clean_x_0 = scheduler.get_velocity(model_output, noisy_coarse_video_latents, timesteps)

        alphas_cumprod = scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(clean_x_0.shape):
            weights = weights.unsqueeze(-1)
        diffusion_loss = torch.mean((weights * (clean_x_0 - real_image) ** 2).reshape(batch_size, -1), dim=1).mean()

        # self.fake_unet.disable_gradient_checkpointing()

        return diffusion_loss

    def compute_guidance_clean_cls_loss(
            self, real_image, fake_image, 
            coarse_image_latents,
            image_latent,
            prompt_embeds,
            image_rotary_emb,

            scheduler,
        ):
        pred_realism_on_real = self.compute_cls_logits(
            real_image.detach(), 
            coarse_image_latents,
            image_latent,
            prompt_embeds,
            image_rotary_emb,

            scheduler,
        )
        pred_realism_on_fake = self.compute_cls_logits(
            fake_image.detach(), 
            coarse_image_latents,
            image_latent,
            prompt_embeds,
            image_rotary_emb,

            scheduler,
        )

        gan_loss = F.softplus(pred_realism_on_fake).mean() + F.softplus(-pred_realism_on_real).mean()
       
        return gan_loss

    def compute_clean_x_0(self,
                        coarse_video_latents,
                        coarse_image_latents,
                        image_latent,
                        prompt_embeds,
                        image_rotary_emb,
                        scheduler,
                        timesteps=None,
                        noise=None,
                    ):
        batch_size = coarse_video_latents.shape[0]
        weight_dtype = coarse_image_latents.dtype
        if timesteps is None:
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (batch_size,), device=coarse_video_latents.device
            )
            timesteps = timesteps.long()
        if noise is None:
            noise = torch.randn_like(coarse_video_latents).to(weight_dtype)
        noisy_coarse_video_latents = scheduler.add_noise(coarse_video_latents, noise, timesteps)
        noisy_model_input = torch.cat([noisy_coarse_video_latents, coarse_image_latents], dim=2)
        model_output = self.feedforward_model(
            hidden_states=noisy_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            image_rotary_emb=image_rotary_emb,
            return_dict=False,
            image_latent = image_latent,
        )[0]
        clean_x_0 = scheduler.get_velocity(model_output, noisy_coarse_video_latents, timesteps) # student clean x_0, [0, 1, ..., 12] 
        return clean_x_0

    def forward(
        self,
        clean_x_0,
        continue_video_latents,
        coarse_video_latents,
        coarse_image_latents,
        image_latent,
        prompt_embeds,
        image_rotary_emb,

        real_unet,
        scheduler,

        losses_to_compute = [],
        timesteps=None,
        noise=None,
    ):    
        real_image = coarse_video_latents
        fake_image = clean_x_0

        loss_dict = {}

        if 'clean_x_0' in losses_to_compute:
            loss_dict['clean_x_0'] = self.compute_clean_x_0(coarse_video_latents,coarse_image_latents,image_latent,prompt_embeds,image_rotary_emb,scheduler, timesteps=timesteps, noise=noise)
        if 'distill_loss' in losses_to_compute:
            loss_dict['distill_loss'] = self.compute_distribution_matching_loss(fake_image, continue_video_latents,coarse_image_latents,image_latent,prompt_embeds,image_rotary_emb,real_unet,scheduler)
        if 'gen_cls_loss' in losses_to_compute:    
            loss_dict['gen_cls_loss'] = self.compute_generator_clean_cls_loss(fake_image,coarse_image_latents,image_latent,prompt_embeds,image_rotary_emb,scheduler)
        if 'diffusion_loss' in losses_to_compute:
            loss_dict['diffusion_loss'] = self.compute_loss_fake(fake_image, real_image,coarse_image_latents,image_latent,prompt_embeds,image_rotary_emb,scheduler)
        if 'gan_loss' in losses_to_compute:
            loss_dict['gan_loss'] = self.compute_guidance_clean_cls_loss(real_image, fake_image, coarse_image_latents, image_latent,prompt_embeds,image_rotary_emb,scheduler)

        return loss_dict