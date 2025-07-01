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

import wandb
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from model.x_mobility.utils import pack_sequence_dim, unpack_sequence_dim
from model.dataset.lerobot_semantic_label import LEROBOT_SEMANTIC_COLORS
from model.dataset.isaac_sim_semantic_label import SEMANTIC_COLORS

def visualise_semantic(batch: Dict, output: Dict) -> torch.Tensor:
    '''Visualizes the semantic segmentation groundtruch and prediction

    Args:
        batch (Dict): input batch
        output (Dict): model predicted output

    Returns:
        viz_video(Tensor): tensor contains the sequence of semantic segmentations
    '''
    # Extract the target and prediction labels [b, s, h, w]

    pred = torch.argmax(output['semantic_segmentation_1'].detach(), dim=-3)
    target = batch['semantic_label'].reshape_as(pred).detach() # ensure target is the same size as pred
    # Create color map [3,3]
    color_map = torch.tensor(SEMANTIC_COLORS,
                             dtype=torch.uint8,
                             device=pred.device)
    # Broadcast to shape [b, s, h, w, 3]
    target = color_map[target.int()]
    pred = color_map[pred.int()]
    # Rearrange the rgb channel [b, s, 3, h, w]
    target = target.permute(0, 1, 4, 2, 3)
    pred = pred.permute(0, 1, 4, 2, 3)
    # Create the concated tensor for viz.
    viz_video = torch.cat([target, pred], dim=-1).detach()
    return viz_video


def visualise_rgb(batch: Dict, output: Dict) -> torch.Tensor:
    '''Visualizes the rgb groundtruch and prediction

    Args:
        batch (Dict): input batch
        output (Dict): model predicted output

    Returns:
        viz_video(Tensor): tensor contains the sequence of rgb predictions
    '''
    # Extract the target and prediction labels [b, s, h, w]
    target = torch.clamp(torch.round(batch['image'] * 255.0), 0,
                         255).to(torch.uint8)
    pred = torch.clamp(torch.round(output['rgb_1'].detach() * 255.0), 0,
                       255).to(torch.uint8)
    # Create the concated tensor for viz.
    viz_video = torch.cat([target, pred], dim=-1).detach()
    return viz_video


def visualise_depth(batch: Dict, output: Dict) -> torch.Tensor:
    '''Visualizes the rgb groundtruch and depthprediction

    Args:
        batch (Dict): input batch
        output (Dict): model predicted output

    Returns:
        viz_video(Tensor): tensor contains the sequence of rgb predictions
    '''
    # Extract the target and prediction labels [b, s, h, w]
    target = torch.clamp(torch.round(batch['image'] * 255.0), 0,
                         255).to(torch.uint8)
    b, s = output['depth'].shape[:2]
    depth_pred = pack_sequence_dim(output['depth'].detach())
    depth_viz = torch.nn.functional.interpolate(
        depth_pred.unsqueeze(1),
        size=batch['image'].shape[-2:],
        mode="bicubic",
        align_corners=False,
    )
    depth_viz = unpack_sequence_dim(depth_viz, b, s)
    depth_viz = torch.clamp(
        torch.round(depth_viz * 255.0 / torch.max(depth_viz)), 0,
        255).to(torch.uint8)
    depth_viz = depth_viz.repeat(1, 1, 3, 1, 1)
    # Create the concated tensor for viz.
    viz_video = torch.cat([target, depth_viz], dim=-1).detach()
    return viz_video


def visualise_attention(batch: Dict, output: Dict):
    '''Visualizes the attentions from image encoder.

    Args:
        batch (Dict): input batch
        output (Dict): model predicted output

    Returns:
        image_list(List): list of attention images.
    '''
    # Pick the first frame of each sample to reduce memory usage.
    image = np.clip(
        batch['image'][:, 0].permute(0, 2, 3, 1).cpu().numpy() * 255, 0,
        255).astype(np.uint8)
    attention_pred = output['image_attentions'].detach()[:, 0].cpu().numpy()
    attention_pred = (attention_pred - np.min(attention_pred)) / (
        np.max(attention_pred) - np.min(attention_pred))
    attention_pred = np.kron(attention_pred, np.ones(
        (14, 14)))  # patch_size=14

    image_list = []
    for img, att in zip(image, attention_pred):
        cmap = plt.get_cmap('jet')
        att_viz = cmap(att)
        att_viz = Image.fromarray(
            (att_viz[:, :, :3] * 255).astype(np.uint8)).convert('RGBA')

        img_viz = Image.fromarray(img).convert('RGBA')
        img_viz = img_viz.resize(att_viz.size[:2], resample=Image.BICUBIC)

        overlayed = Image.blend(img_viz, att_viz, alpha=0.5)
        image_list.append(wandb.Image(overlayed))
    return image_list
