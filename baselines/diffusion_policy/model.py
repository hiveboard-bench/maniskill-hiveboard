# Defines the diffusion policy model.
# Created: 2026-09-14

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from transformers import CLIPVisionModel

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    # Compute the forward pass.
    def forward(self, x):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -(math.log(10000) / (half_dim - 1)))
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    # Compute the forward pass.
    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two conv blocks with FiLM conditioning (per-channel scale and bias predicted from the condition)."""

    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out_channels * 2))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    # Compute the forward pass.
    def forward(self, x, cond):
        out = self.blocks[0](x)
        scale, bias = self.cond_encoder(cond).reshape(cond.shape[0], 2, self.out_channels, 1).unbind(1)
        out = self.blocks[1](scale * out + bias)
        return out + self.residual_conv(x)


class ConditionalUnet1D(nn.Module):
    """1D temporal U-Net over action sequences, conditioned globally on the diffusion step and observation features."""

    def __init__(self, input_dim, global_cond_dim, diffusion_step_embed_dim=256, down_dims=(256, 512, 1024), kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim, *down_dims]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed)
        )
        cond_dim = dsed + global_cond_dim
        block = lambda i, o: ConditionalResidualBlock1D(i, o, cond_dim, kernel_size, n_groups)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList([
            nn.ModuleList([block(i, o), block(o, o), nn.Conv1d(o, o, 3, 2, 1) if k < len(in_out) - 1 else nn.Identity()])
            for k, (i, o) in enumerate(in_out)
        ])
        self.mid_modules = nn.ModuleList([block(all_dims[-1], all_dims[-1]), block(all_dims[-1], all_dims[-1])])
        self.up_modules = nn.ModuleList([
            nn.ModuleList([block(o * 2, i), block(i, i), nn.ConvTranspose1d(i, i, 4, 2, 1) if k < len(in_out) - 1 else nn.Identity()])
            for k, (i, o) in enumerate(reversed(in_out[1:]))
        ])
        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size), nn.Conv1d(down_dims[0], input_dim, 1)
        )

    # Compute the forward pass.
    def forward(self, sample, timestep, global_cond):
        x = sample.transpose(1, 2)
        cond = torch.cat([self.diffusion_step_encoder(timestep.expand(sample.shape[0])), global_cond], dim=-1)
        skips = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet2(resnet(x, cond), cond)
            skips.append(x)
            x = downsample(x)
        for mid in self.mid_modules:
            x = mid(x, cond)
        for resnet, resnet2, upsample in self.up_modules:
            x = resnet(torch.cat((x, skips.pop()), dim=1), cond)
            x = upsample(resnet2(x, cond))
        return self.final_conv(x).transpose(1, 2)


class EMAModel:
    """Exponential moving average of weights with warmup (decay = 1 - (1 + step / inv_gamma) ** -power)."""

    def __init__(self, model, inv_gamma=1.0, power=0.75, max_value=0.9999):
        self.averaged_model = copy.deepcopy(model).eval().requires_grad_(False)
        self.inv_gamma, self.power, self.max_value = inv_gamma, power, max_value
        self.optimization_step = 0

    # Update the exponential moving average of model weights.
    @torch.no_grad()
    def step(self, model):
        step = max(0, self.optimization_step - 1)
        decay = 0.0 if step <= 0 else min(1 - (1 + step / self.inv_gamma) ** -self.power, self.max_value)
        for module, ema_module in zip(model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                if isinstance(module, nn.modules.batchnorm._BatchNorm) or not param.requires_grad:
                    ema_param.copy_(param.data)
                else:
                    ema_param.mul_(decay).add_(param.data, alpha=1 - decay)
            for buffer, ema_buffer in zip(module.buffers(recurse=False), ema_module.buffers(recurse=False)):
                ema_buffer.copy_(buffer)
        self.optimization_step += 1


# Swap BatchNorm layers for GroupNorm so small batches train reliably.
def replace_batchnorm_with_groupnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(child.num_features // 16, child.num_features))
        else:
            replace_batchnorm_with_groupnorm(child)
    return module


class ResNet18Encoder(nn.Module):
    """ResNet-18 trunk with global average pooling; GroupNorm replaces BatchNorm when trained from scratch."""

    def __init__(self, pretrained):
        super().__init__()
        net = torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        net.fc = nn.Identity()
        self.net = net if pretrained else replace_batchnorm_with_groupnorm(net)
        self.out_dim = 512
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    # Compute the forward pass.
    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class CLIPEncoder(nn.Module):
    """CLIP vision transformer; the pooled CLS embedding is the image feature."""

    def __init__(self, model_name):
        super().__init__()
        self.net = CLIPVisionModel.from_pretrained(model_name)
        self.net.gradient_checkpointing_enable()
        self.out_dim = self.net.config.hidden_size
        self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)

    # Compute the forward pass.
    def forward(self, x):
        return self.net(pixel_values=(x - self.mean) / self.std).pooler_output


class ObsEncoder(nn.Module):
    """Per-camera image features (random crop in training, center crop in eval) concatenated with the robot state."""

    def __init__(self, cameras, encoder, input_size, crop_ratio, clip_model=None):
        super().__init__()
        self.cameras = cameras
        self.input_size = input_size
        self.crop_ratio = crop_ratio
        if encoder == "clip":
            shared = CLIPEncoder(clip_model)
            self.backbones = nn.ModuleDict({"shared": shared})
            self.backbone_for = {camera: "shared" for camera in cameras}
        else:
            self.backbones = nn.ModuleDict({camera: ResNet18Encoder(pretrained=encoder == "imagenet") for camera in cameras})
            self.backbone_for = {camera: camera for camera in cameras}
        self.image_feature_dim = next(iter(self.backbones.values())).out_dim

    # Uint8 (N, H, W, 3) -> float (N, 3, input_size, input_size) in [0, 1] via one affine crop and resize.
    def crop_resize(self, images):
        x = images.permute(0, 3, 1, 2).float() / 255
        n = x.shape[0]
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0] = theta[:, 1, 1] = self.crop_ratio
        if self.training:
            theta[:, :, 2] = (torch.rand(n, 2, device=x.device) * 2 - 1) * (1 - self.crop_ratio)
        grid = F.affine_grid(theta, (n, 3, self.input_size, self.input_size), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", align_corners=False)

    # Compute the forward pass.
    def forward(self, images, state):
        features = [self.backbones[self.backbone_for[camera]](self.crop_resize(images[camera])) for camera in self.cameras]
        return torch.cat([*features, state], dim=-1)


class DiffusionPolicy(nn.Module):
    """Predicts an action sequence by denoising, conditioned on the last n_obs_steps observations."""

    def __init__(self, cameras, state_dim, action_dim, stats, encoder, policy_config, encoder_config):
        super().__init__()
        self.horizon = policy_config["horizon"]
        self.n_obs_steps = policy_config["n_obs_steps"]
        self.n_action_steps = policy_config["n_action_steps"]
        self.action_dim = action_dim
        self.obs_encoder = ObsEncoder(
            cameras, encoder, encoder_config["input_size"], policy_config["crop_ratio"], encoder_config.get("model")
        )
        obs_feature_dim = self.obs_encoder.image_feature_dim * len(cameras) + state_dim
        self.model = ConditionalUnet1D(
            input_dim=action_dim,
            global_cond_dim=obs_feature_dim * self.n_obs_steps,
            diffusion_step_embed_dim=policy_config["diffusion_step_embed_dim"],
            down_dims=policy_config["down_dims"],
            kernel_size=policy_config["kernel_size"],
            n_groups=policy_config["n_groups"],
        )
        self.scheduler_config = dict(
            num_train_timesteps=policy_config["num_train_timesteps"],
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        self.scheduler_config.update(set_alpha_to_one=True, steps_offset=0)
        self.noise_scheduler = DDIMScheduler(**self.scheduler_config)
        self.num_inference_steps = policy_config["num_inference_steps"]
        for name, value in stats.items():
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))

    # Scale a recorded value to [-1, 1] using training bounds.
    def normalize(self, x, key):
        low, high = getattr(self, f"{key}_min"), getattr(self, f"{key}_max")
        varies = high - low > 1e-4
        scale = torch.where(varies, high - low, torch.ones_like(high))
        return torch.where(varies, (x - low) / scale * 2 - 1, torch.zeros_like(x))

    def unnormalize(self, x, key):
        low, high = getattr(self, f"{key}_min"), getattr(self, f"{key}_max")
        varies = high - low > 1e-4
        scale = torch.where(varies, high - low, torch.ones_like(high))
        return torch.where(varies, (x + 1) / 2 * scale + low, low.expand_as(x))

    # Build the global conditioning input.
    def global_cond(self, images, state):
        batch = state.shape[0]
        flat_images = {camera: frames[:, : self.n_obs_steps].flatten(0, 1) for camera, frames in images.items()}
        flat_state = self.normalize(state[:, : self.n_obs_steps], "state").flatten(0, 1)
        return self.obs_encoder(flat_images, flat_state).reshape(batch, -1)

    # Train the denoiser to predict the noise added to action sequences.
    def compute_loss(self, batch):
        cond = self.global_cond(batch["images"], batch["state"])
        actions = self.normalize(batch["action"], "action")
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (actions.shape[0],), device=actions.device)
        noisy = self.noise_scheduler.add_noise(actions, noise, timesteps)
        return F.mse_loss(self.model(noisy, timesteps, cond), noise)

    # Select a supported inference scheduler and step count.
    def configure_denoising(self, scheduler_type="ddim", inference_steps=None):
        scheduler_type = scheduler_type.lower()
        if scheduler_type == "ddim":
            self.noise_scheduler = DDIMScheduler(**self.scheduler_config)
        elif scheduler_type == "ddpm":
            config = {key: value for key, value in self.scheduler_config.items()
                      if key not in ("set_alpha_to_one", "steps_offset")}
            self.noise_scheduler = DDPMScheduler(**config)
        else:
            raise ValueError(f"unsupported scheduler: {scheduler_type}")
        if inference_steps is not None:
            if inference_steps <= 0:
                raise ValueError("inference_steps must be positive")
            self.num_inference_steps = int(inference_steps)
        return scheduler_type

    # Return the normalized full-horizon sample and observation condition.
    def sample_trajectory(self, images, state, guidance=None, capture_x0=False, ddim_eta=0.0, variance_steps=4):
        with torch.no_grad():
            cond = self.global_cond(images, state)
        trajectory = torch.randn(state.shape[0], self.horizon, self.action_dim, device=state.device)
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        x0_estimates = []
        if capture_x0 and variance_steps > self.num_inference_steps:
            raise ValueError("variance_steps cannot exceed num_inference_steps")
        for step, t in enumerate(self.noise_scheduler.timesteps):
            with torch.no_grad():
                output = self.model(trajectory, t.to(state.device), cond)
            if guidance is not None:
                output = guidance(trajectory, output, cond, step, t)
            with torch.no_grad():
                if isinstance(self.noise_scheduler, DDIMScheduler):
                    scheduler_output = self.noise_scheduler.step(output, t, trajectory, eta=ddim_eta)
                else:
                    scheduler_output = self.noise_scheduler.step(output, t, trajectory)
                trajectory = scheduler_output.prev_sample
                if capture_x0:
                    x0_estimates.append(scheduler_output.pred_original_sample.detach())
                    if len(x0_estimates) > variance_steps:
                        x0_estimates.pop(0)
        if capture_x0:
            estimates = torch.stack(x0_estimates, dim=0)
            variance = estimates.var(dim=0, unbiased=False).sum(dim=-1)
            return trajectory, cond, variance
        return trajectory, cond

    # Convert a normalized horizon sample to the executable action chunk.
    def actions_from_trajectory(self, trajectory):
        actions = self.unnormalize(trajectory, "action")
        start = self.n_obs_steps - 1
        return actions[:, start : start + self.n_action_steps]

    # Predict the executable action chunk; prompt is unused.
    @torch.no_grad()
    def predict_action(self, images, state, prompt=None):
        trajectory, _ = self.sample_trajectory(images, state)
        return self.actions_from_trajectory(trajectory)

    # Predict an executable chunk and return its observation condition for offline analysis.
    @torch.no_grad()
    def predict_action_with_features(self, images, state, prompt=None):
        del prompt
        trajectory, condition = self.sample_trajectory(images, state)
        return self.actions_from_trajectory(trajectory), condition

    # Predict an unchanged action chunk and its normalized-space denoising variance.
    @torch.no_grad()
    def predict_action_with_variance(self, images, state, prompt=None, variance_steps=4, ddim_eta=0.0):
        del prompt
        trajectory, _, variance = self.sample_trajectory(
            images, state, capture_x0=True, ddim_eta=ddim_eta, variance_steps=variance_steps,
        )
        return self.actions_from_trajectory(trajectory), variance

    # Extract final-denoising hidden states before the action prediction head.
    @torch.no_grad()
    def action_head_features_from_condition(self, condition, generator):
        trajectory = torch.randn(condition.shape[0], self.horizon, self.action_dim,
                                 device=condition.device, generator=generator)
        captured = {}
        handle = self.model.final_conv[0].register_forward_hook(
            lambda _module, _inputs, output: captured.update(feature=output.detach()))
        try:
            self.noise_scheduler.set_timesteps(self.num_inference_steps)
            for index, timestep in enumerate(self.noise_scheduler.timesteps):
                output = self.model(trajectory, timestep.to(condition.device), condition)
                if index + 1 == len(self.noise_scheduler.timesteps):
                    features = captured["feature"].transpose(1, 2)
                trajectory = self.noise_scheduler.step(output, timestep, trajectory).prev_sample
        finally:
            handle.remove()
        start = self.n_obs_steps - 1
        return features[:, start:start + self.n_action_steps]
