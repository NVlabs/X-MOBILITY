# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

import gin
import torch
import torch.nn.functional as F
import torchvision.models
from torch import nn

from model.x_mobility.utils import pack_sequence_dim


@gin.configurable
class SegmentationLoss(nn.Module):
    '''Loss for semantic segmentaiton

        Inputs:
            prediction: predicted semantic image (b, s, c_s, h, w)
            target: ground-truth semantic image (b, s, h, w)

        Returns:
            loss: float
    '''
    def __init__(self, use_top_k: bool, top_k_ratio: float, use_poly_one: bool,
                 poly_one_coefficient: float, use_weights: bool,
                 semantic_weights: list):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.use_weights = use_weights
        self.use_poly_one = use_poly_one
        self.poly_one_coefficient = poly_one_coefficient

        if self.use_weights:
            self.weights = semantic_weights

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        b, s, c, h, w = prediction.shape
        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w).long()

        weights = torch.tensor(
            self.weights, dtype=prediction.dtype,
            device=prediction.device) if self.use_weights else None

        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=weights,
        )
        loss = loss[~torch.isnan(loss)]
        if self.use_poly_one:
            prob = torch.exp(-loss)
            loss_poly_one = self.poly_one_coefficient * (1 - prob)
            loss = loss + loss_poly_one
        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss = loss.topk(k, dim=-1)[0]

        return torch.mean(loss)


class PerceptualLoss(nn.Module):
    '''Perceptural loss for RGB reconstruction.
       Ref paper: Perceptual Losses for Real-Time Style Transfer and Super-Resolution

    Args:
        x: pred image (N, 3, H, W)
        y: target image (N, 3, H, W)
    '''
    def __init__(self):
        super().__init__()

        # VGG blocks
        blocks = []
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(
            torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for block in blocks:
            for p in block.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)

        # Normalization parameters for VGG-16
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        # Normalize input images
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = 0.0
        x = pred
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss


@gin.configurable
class RgbLoss(nn.Module):
    '''Loss for RGB prediction

    Inputs:
        prediction: predicted rgb image (b, s, 3, h, w)
        target: ground-truth rgb image (b, s, 3, h, w)

    Returns:
        loss: loss for RGB
    '''
    def __init__(self):
        super().__init__()
        self.perceptual_loss = PerceptualLoss()
        self.l1_loss = F.l1_loss

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        assert len(prediction.shape) == 5, 'Prediction must be a 5D tensor'
        l1_loss = self.l1_loss(prediction, target, reduction='none')
        l1_loss = torch.sum(l1_loss, dim=-3, keepdims=True).mean()
        perceptual_loss = self.perceptual_loss(pack_sequence_dim(prediction),
                                               pack_sequence_dim(target))
        return l1_loss + perceptual_loss


@gin.configurable
class ActionLoss(nn.Module):
    '''Loss for action regression

        Args:
            norm: decides L1 or L2 loss.

        Inputs:
            prediction: predicted action (b, s, c_a)
            target: ground-truth action (b, s, c_a)

        Returns:
            loss: loss for action
    '''
    def __init__(self, norm, channel_dim=-1):
        super().__init__()
        self.norm = norm
        self.channel_dim = channel_dim

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=self.channel_dim, keepdims=True)
        return loss.mean()


class PathLoss(nn.Module):
    '''Loss for path regression

        Inputs:
            prediction: predicted action (b, s, c_p)
            target: ground-truth action (b, s, c_p)

        Returns:
            loss: loss for path
    '''
    def __init__(self, channel_dim=-1):
        super().__init__()
        self.channel_dim = channel_dim
        self.loss_fn = F.mse_loss

    def forward(self, prediction: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=self.channel_dim, keepdims=True)
        return loss.mean()


class ProbabilisticLoss(nn.Module):
    ''' KL divergence loss between a prior distribution and a posterior distribution

        Inputs:
            prior_mu, prior_sigma: Prior distributions
            posterior_mu, posterior_sigma: Poserior distributions

        Returns:
            loss: KL divergence between the two distributions.
    '''
    def forward(self, prior_mu: torch.Tensor, prior_sigma: torch.Tensor,
                posterior_mu: torch.Tensor,
                posterior_sigma: torch.Tensor) -> torch.Tensor:
        posterior_var = posterior_sigma[:, 1:]**2
        prior_var = prior_sigma[:, 1:]**2

        posterior_log_sigma = torch.log(posterior_sigma[:, 1:])
        prior_log_sigma = torch.log(prior_sigma[:, 1:])

        kl_div = (prior_log_sigma - posterior_log_sigma - 0.5 +
                  (posterior_var +
                   (posterior_mu[:, 1:] - prior_mu[:, 1:])**2) /
                  (2 * prior_var))
        first_kl = -posterior_log_sigma[:, :1] - 0.5 + (
            posterior_var[:, :1] + posterior_mu[:, :1]**2) / 2
        kl_div = torch.cat([first_kl, kl_div], dim=1)

        # Sum across channel dimension
        # Average across batch dimension, keep time dimension for monitoring
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss


@gin.configurable
class KLLoss(nn.Module):
    ''' Balanced loss for KL divergence

        Inputs:
            prior: Prior distributions
            posterio: Poserior distributions

        Returns:
            loss: Balanced KL divergence between the two distributions.
    '''
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss = ProbabilisticLoss()

    def forward(self, prior: Dict, posterior: Dict) -> float:
        prior_mu, prior_sigma = prior['mu'], prior['sigma']
        posterior_mu, posterior_sigma = posterior['mu'], posterior['sigma']
        prior_loss = self.loss(prior_mu, prior_sigma, posterior_mu.detach(),
                               posterior_sigma.detach())
        posterior_loss = self.loss(prior_mu.detach(), prior_sigma.detach(),
                                   posterior_mu, posterior_sigma)

        return self.alpha * prior_loss + (1 - self.alpha) * posterior_loss


@gin.configurable
class DiffusionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = F.mse_loss

    def forward(self, noise: torch.tensor, noise_pred: torch.tensor) -> float:
        loss = self.loss(noise, noise_pred)
        return loss


@gin.configurable
class DepthLoss(nn.Module):
    ''' Loss for depth prediction.
    '''
    def __init__(self, alpha=1.0, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()

    def forward(self, predicted, target):
        # L1 loss
        l1_loss = self.l1_loss(predicted, target)

        # Gradient loss
        gradient_loss = self.compute_gradient_loss(predicted, target)

        # Combined loss
        return self.alpha * l1_loss + self.beta * gradient_loss

    def compute_gradient_loss(self, predicted, target):
        pred_dx = torch.abs(predicted[:, :, :-1] - predicted[:, :, 1:])
        pred_dy = torch.abs(predicted[:, :-1, :] - predicted[:, 1:, :])
        target_dx = torch.abs(target[:, :, :-1] - target[:, :, 1:])
        target_dy = torch.abs(target[:, :-1, :] - target[:, 1:, :])

        grad_loss_x = self.l1_loss(pred_dx, target_dx)
        grad_loss_y = self.l1_loss(pred_dy, target_dy)

        gradient_loss = grad_loss_x + grad_loss_y
        return gradient_loss


@gin.configurable
class XMobilityLoss(nn.Module):
    ''' Aggregated loss for X-mobility model trianing

        Inputs:
            output (dict): dict of model outputs
                action: (b, s, c_a)
                prior: dict of prior state estimator
                posterior: dict of posterior state estimator
                semantic_segmentation_1:  (b, s, c_semantic, h, w)
                rgb_1: (b, s, 3, h, w)
            batch (Dict): dict of the target tensors:
                action: (b, s, c_a)
                semantic_label: (b, s, h, w)

        Returns:
            losses: dict of loss items (action, kl, semantic_segmentation)
    '''
    def __init__(self,
                 action_weight: float,
                 path_weight: float,
                 kl_weight: float,
                 semantic_weight: float,
                 rgb_weight: float,
                 diffusion_weight: float,
                 depth_weight: float,
                 enable_semantic: bool,
                 enable_rgb_stylegan: bool,
                 enable_rgb_diffusion: bool,
                 enable_policy_diffusion: bool,
                 is_gwm_pretrain: bool = False):
        super().__init__()
        self.action_weight = action_weight
        self.path_weight = path_weight
        self.kl_weight = kl_weight
        self.semantic_weight = semantic_weight
        self.rgb_weight = rgb_weight
        self.diffusion_weight = diffusion_weight
        self.depth_weight = depth_weight
        self.enable_semantic = enable_semantic
        self.enable_rgb_stylegan = enable_rgb_stylegan
        self.enable_rgb_diffusion = enable_rgb_diffusion
        self.enable_policy_diffusion = enable_policy_diffusion
        self.is_gwm_pretrain = is_gwm_pretrain

        if not self.is_gwm_pretrain:
            if self.enable_policy_diffusion:
                self.policy_diffusion_loss = DiffusionLoss()
            else:
                self.action_loss = ActionLoss()
                self.path_loss = PathLoss()

        self.kl_loss = KLLoss()

        if self.enable_semantic:
            self.segmentation_loss = SegmentationLoss()
        if self.enable_rgb_stylegan:
            self.rgb_loss = RgbLoss()
        if self.enable_rgb_diffusion:
            self.diffusion_loss = DiffusionLoss()

        self.depth_loss = DepthLoss()

    def forward(self, output: Dict, batch: Dict) -> Dict:
        losses = {}

        if not self.is_gwm_pretrain:
            if self.enable_policy_diffusion:
                losses[
                    'action'] = self.action_weight * self.policy_diffusion_loss(
                        output['action_noise'], output['action_noise_pred'])
                losses['path'] = self.path_weight * self.policy_diffusion_loss(
                    output['path_noise'], output['path_noise_pred'])
            else:
                losses['action'] = self.action_weight * self.action_loss(
                    output['action'], batch['action'])
                if 'path' in output and 'path' in batch:
                    losses['path'] = self.path_weight * self.path_loss(
                        output['path'], batch['path'])

        losses['kl'] = self.kl_weight * self.kl_loss(output['prior'],
                                                     output['posterior'])

        # Semantic segmentation loss.
        if self.enable_semantic:
            for downsampling_factor in [1, 2, 4]:
                if f"semantic_segmentation_{downsampling_factor}" not in output:
                    continue
                semantic_segmentation_loss = self.segmentation_loss(
                    prediction=output[
                        f"semantic_segmentation_{downsampling_factor}"],
                    target=batch[f"semantic_label_{downsampling_factor}"],
                )
                discount = 1 / downsampling_factor
                losses[f"semantic_segmentation_{downsampling_factor}"] = (
                    discount * self.semantic_weight *
                    semantic_segmentation_loss)

        # StyleGan RGB regression loss.
        if self.enable_rgb_stylegan:
            for downsampling_factor in [1, 2, 4]:
                if f"rgb_{downsampling_factor}" not in output:
                    continue
                discount = 1 / downsampling_factor
                rgb_loss = self.rgb_loss(
                    prediction=output[f"rgb_{downsampling_factor}"],
                    target=batch[f"rgb_label_{downsampling_factor}"],
                )
                losses[
                    f"rgb_{downsampling_factor}"] = discount * self.rgb_weight * rgb_loss

        # Diffusion RGB loss.
        if self.enable_rgb_diffusion:
            losses['diffusion'] = self.diffusion_weight * self.diffusion_loss(
                output['rgb_noise'], output['rgb_noise_pred'])

        # Depth loss.
        if 'depth' in output and 'depth_gt' in output:
            losses['depth'] = self.depth_weight * self.depth_loss(
                output['depth'], output['depth_gt'])

        return losses
