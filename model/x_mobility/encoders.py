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
from torch import nn
from transformers import DepthAnythingForDepthEstimation, Dinov2Model

from model.x_mobility.utils import pack_sequence_dim, unpack_sequence_dim

# Need to compute this dynamically and adapt to different image size.
DEPTH_ANYTHING_IMAGE_SIZE = [322, 518]


@gin.configurable
class SpeedEncoder(nn.Module):
    '''Encoder of robot's speed

        Args:
            out_channels (int): output channels size
            speed_normalisation (float): speed normialisation factor
    '''
    def __init__(self, out_channels: int, speed_normalisation: float):
        super().__init__()
        self.speed_encoder = nn.Sequential(
            nn.Linear(1, out_channels),
            nn.ReLU(True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(True),
        )
        self.out_channels = out_channels
        self.speed_normalisation = speed_normalisation

    def forward(self, speed: torch.Tensor) -> torch.Tensor:
        return self.speed_encoder(speed / self.speed_normalisation)


@gin.configurable
class ImageDINOEncoder(nn.Module):
    ''' Image encoding with DINO v2
    '''
    def __init__(self, enable_fine_tune=False):
        super().__init__()
        self.dino_model = Dinov2Model.from_pretrained('facebook/dinov2-small',
                                                      output_attentions=True)
        # Freeze the pre-trained dino model.
        if not enable_fine_tune:
            for param in self.dino_model.parameters():
                param.requires_grad = False
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_channels = self.dino_model.config.hidden_size * 2

    def forward(self, image):
        # Preprocess image.
        dino_image_size = self.dino_model.config.image_size
        processed_image = F.interpolate(
            image,
            size=[dino_image_size, dino_image_size],
            mode='bicubic',
            align_corners=False)
        processed_image = (processed_image - self.mean) / self.std

        # Encode image.
        outputs = self.dino_model(processed_image)

        # Extract features.
        last_hidden_states = outputs[0]
        cls_token = last_hidden_states[:, 0]
        patch_tokens = last_hidden_states[:, 1:]
        features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        # Extract attentions
        n, _, h, w = processed_image.shape
        avg_attension = torch.mean(outputs.attentions[-1], dim=1)
        cls_attention = avg_attension[:, 0, 1:].view(n, h // 14, w // 14)

        return {'image_features': features, 'image_attentions': cls_attention}


@gin.configurable
class ImageDepthAnythingEncoder(nn.Module):
    ''' Image encoding with Depth Anything.
    '''
    def __init__(self, enable_fine_tune=False):
        super().__init__()
        self.enable_fine_tune = enable_fine_tune
        self.depth_model = DepthAnythingForDepthEstimation.from_pretrained(
            "LiheYoung/depth-anything-small-hf",
            output_hidden_states=True,
            output_attentions=True)
        if not enable_fine_tune:
            for param in self.depth_model.parameters():
                param.requires_grad = False
        else:
            self.depth_gt_model = DepthAnythingForDepthEstimation.from_pretrained(
                "LiheYoung/depth-anything-small-hf",
                output_hidden_states=False,
                output_attentions=False)
            for param in self.depth_gt_model.parameters():
                param.requires_grad = False
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.out_channels = self.depth_model.backbone.config.hidden_size * 2

    def forward(self, image):
        # Preprocess image.
        processed_image = F.interpolate(image,
                                        size=DEPTH_ANYTHING_IMAGE_SIZE,
                                        mode='bicubic',
                                        align_corners=False)
        processed_image = (processed_image - self.mean) / self.std

        # Encode image.
        outputs = self.depth_model(processed_image)

        # Extract features.
        last_hidden_states = outputs.hidden_states[-1]
        cls_token = last_hidden_states[:, 0]
        patch_tokens = last_hidden_states[:, 1:]
        features = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)

        # Extract attentions
        n, _, h, w = processed_image.shape
        avg_attension = torch.mean(outputs.attentions[-1], dim=1)
        cls_attention = avg_attension[:, 0, 1:].view(n, h // 14, w // 14)

        ret_dict = {
            'image_features': features,
            'image_attentions': cls_attention,
            'depth': outputs.predicted_depth,
        }

        # Get depth GT based on the pretrained DepthAnything model.
        if self.enable_fine_tune:
            depth_gt = self.depth_gt_model(processed_image).predicted_depth
            ret_dict['depth_gt'] = depth_gt

        return ret_dict


@gin.configurable
class ObservationEncoder(nn.Module):
    '''Encoder of the input observations

        Inputs:
            batch (Dict): dict of the input tensors:
                image: (b, s, 3, h, w)
                speed: (b, s, 1)
                action: (b, s, 6)

        Returns:
            embedding: 1D embedding of the observations
    '''
    def __init__(self, image_encoder: nn.Module):
        super().__init__()

        # Image
        self.image_encoder = image_encoder()
        features_channels = self.image_encoder.out_channels

        # Speed.
        self.speed_encoder = SpeedEncoder()
        features_channels += self.speed_encoder.out_channels

        self.embedding_dim = features_channels

    def forward(self, batch: Dict) -> torch.Tensor:
        b, s = batch['image'].shape[:2]
        image = pack_sequence_dim(batch['image'])
        speed = pack_sequence_dim(batch['speed'])

        # Image encoding
        image_encoding_outputs = self.image_encoder(image)

        # Speed encoding
        speed_features = self.speed_encoder(speed)

        # Final observation embedding.
        embedding = torch.cat(
            [image_encoding_outputs['image_features'], speed_features], dim=1)

        # Compose outputs.
        outputs = {}
        outputs['embedding'] = unpack_sequence_dim(embedding, b, s)
        outputs['speed_features'] = unpack_sequence_dim(speed_features, b, s)
        for k, v in image_encoding_outputs.items():
            outputs[k] = unpack_sequence_dim(v, b, s)
        return outputs
