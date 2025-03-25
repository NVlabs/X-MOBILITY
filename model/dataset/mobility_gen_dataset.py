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

import bisect
import io
import os

import gin
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as tvf
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from scipy.spatial.transform import Rotation

from model.dataset.semantic_label import SemanticLabel
from model.dataset.data_constants import INPUT_IMAGE_SIZE

# Sim labels mapping to group semantic classes as needed.
SIM_LABELS_MAPPING = {
    'floor': SemanticLabel.NAVIGABLE,
    'floor_decal': SemanticLabel.NAVIGABLE,
    'forklift': SemanticLabel.FORKLIFT,
    'pallet': SemanticLabel.PALLET,
    'fence': SemanticLabel.FENCE,
    'hazard_sign,sign': SemanticLabel.SIGN,
    'cone,traffic_cone': SemanticLabel.CONE
}

ROUTE_POSE_SIZE = 2
FIXED_ROUTE_SIZE = 20


# Create a 4x4 transformation matrix from position and quaternion.
def create_transform_matrix(position, quaternion):
    # Create rotation matrix from quaternion
    r = Rotation.from_quat(
        [quaternion[0], quaternion[1], quaternion[2], quaternion[3]])
    rotation_matrix = r.as_matrix()

    # Create transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position

    return transform


# Transform 2D points from world frame to robot frame.
def world_to_robot_frame(world_points, robot_position, robot_quaternion):
    # Create world to robot transform
    world_to_robot = create_transform_matrix(robot_position, robot_quaternion)

    # Invert to get robot to world transform
    robot_to_world = np.linalg.inv(world_to_robot)

    # Convert 2D points to homogeneous coordinates (add z=0 and w=1)
    n_points = len(world_points)
    homogeneous_points = np.ones((n_points, 4))
    homogeneous_points[:, 0:2] = world_points

    # Transform points
    transformed_points = np.zeros((n_points, 2))
    for i in range(n_points):
        transformed_point = robot_to_world @ homogeneous_points[i]
        transformed_points[i] = transformed_point[:2] / transformed_point[3]

    return transformed_points


def interpolate_resize(x, size, mode=tvf.InterpolationMode.NEAREST):
    '''Resize the tensor with interpolation
    '''
    x = tvf.resize(x, size, interpolation=mode, antialias=True)
    return x


@gin.configurable
class XMobilityMobilityGenDataModule(pl.LightningDataModule):
    '''Datamodule with dataset collected from MobilityGen.
    '''

    def __init__(self,
                 dataset_path: str,
                 batch_size: int,
                 sequence_length: int,
                 num_workers: int,
                 enable_semantic: bool = False,
                 enable_rgb_stylegan: bool = False,
                 is_gwm_pretrain: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.enable_semantic = enable_semantic
        self.enable_rgb_stylegan = enable_rgb_stylegan
        self.is_gwm_pretrain = is_gwm_pretrain
        self.dataset_path = dataset_path
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = MobilityGenDataset(
                os.path.join(self.dataset_path, 'train'), self.sequence_length,
                self.enable_semantic, self.enable_rgb_stylegan,
                self.is_gwm_pretrain)
            self.val_dataset = MobilityGenDataset(
                os.path.join(self.dataset_path, 'val'), self.sequence_length,
                self.enable_semantic, self.enable_rgb_stylegan,
                self.is_gwm_pretrain)
        if stage == 'test' or stage is None:
            self.test_dataset = MobilityGenDataset(
                os.path.join(self.dataset_path, 'test'), self.sequence_length,
                self.enable_semantic, self.enable_rgb_stylegan,
                self.is_gwm_pretrain)

    def train_dataloader(self):
        train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          sampler=train_sampler)

    def val_dataloader(self):
        val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          sampler=val_sampler)

    def test_dataloader(self):
        test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          drop_last=True,
                          sampler=test_sampler)

    def load_test_data(self):
        """ Load test data. This function is used in local Jupyter testing environment only.
        """
        test_dataset = MobilityGenDataset(
            os.path.join(self.dataset_path,
                         'test'), self.sequence_length, self.enable_semantic,
            self.enable_rgb_stylegan, self.is_gwm_pretrain)
        return DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )


class MobilityGenDataset(Dataset):
    '''Dataset from Mobility Gen.
    '''

    def __init__(self,
                 dataset_path: str,
                 sequence_length: int,
                 enable_semantic: bool = False,
                 enable_rgb_stylegan: bool = False,
                 is_gwm_pretrain: bool = False):
        super().__init__()
        self.sequence_length = sequence_length
        self.enable_semantic = enable_semantic
        self.enable_rgb_stylegan = enable_rgb_stylegan
        self.is_gwm_pretrain = is_gwm_pretrain
        self.dfs = []
        self.accumulated_sample_sizes = []
        self.num_samples = 0

        # Iterate each scenario in the dataset.
        for scenario in os.listdir(dataset_path):
            scenario_path = os.path.join(dataset_path, scenario)
            # Iterate the sorted runs for the given scenario.
            run_files = [
                run_file for run_file in os.listdir(scenario_path)
                if run_file.endswith('pqt')
            ]
            run_files = sorted(run_files)
            with tqdm(total=len(run_files),
                      desc=f"Loading data from {scenario_path}",
                      unit="file") as pbar:
                for run_file in run_files:
                    parquet_path = os.path.join(scenario_path, run_file)
                    df = pd.read_parquet(parquet_path, engine='pyarrow')
                    self.dfs.append(df)
                    self.accumulated_sample_sizes.append(self.num_samples)
                    self.num_samples += len(df) // self.sequence_length
                    pbar.update(1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        batch = {}
        # Get the cooresponding df.
        df_idx = bisect.bisect_left(self.accumulated_sample_sizes,
                                    index + 1) - 1
        for seq_idx in range(self.sequence_length):
            sample_idx = (index - self.accumulated_sample_sizes[df_idx]
                          ) * self.sequence_length + seq_idx
            element = self._get_element(self.dfs[df_idx], sample_idx)
            for k, v in element.items():
                batch[k] = batch.get(k, []) + [v]
        # Convert np array to tensor
        for k, v in batch.items():
            batch[k] = torch.from_numpy(np.stack(v)).type(torch.float32)

        # Downsample the input images.
        self._down_sample_input_image(batch)

        # Prepare scaled rgb and semantic labels
        if self.enable_rgb_stylegan:
            self._compose_rgb_labels(batch)

        if self.enable_semantic:
            self._compose_semantic_labels(batch)

        return batch

    def _get_element(self, df, sample_index):
        sample = df.iloc[sample_index]
        element = {}
        element['action'] = self._get_action(sample)
        element['image'] = self._get_rgb_image(sample)
        element['speed'] = self._get_speed(sample)
        if self.enable_semantic:
            element['semantic_label'] = self._get_semantic_label(sample)

        if not self.is_gwm_pretrain:
            element['route_vectors'] = self._get_route_vector(sample)
        return element

    def _get_rgb_image(self, sample):
        rgb_image = Image.open(
            io.BytesIO(sample['robot.front_camera.left.rgb_image']))
        return np.transpose(np.array(rgb_image), (2, 0, 1)) / 255.0

    def _get_route_vector(self, sample):
        # MobilityGen does not provide route_poses, so we use the target_path instead.
        if 'target_path' not in sample:
            return None
        route_poses = np.array(sample['target_path'], np.float32)
        route_poses = route_poses.reshape(
            len(route_poses) // ROUTE_POSE_SIZE, ROUTE_POSE_SIZE)
        # Transform the route_poses to the robot frame.
        route_poses = world_to_robot_frame(route_poses,
                                           sample['robot.position'],
                                           sample['robot.orientation'])
        # Truncate or extend the route_poses to the fixed size.
        if len(route_poses) > FIXED_ROUTE_SIZE:
            route_poses = route_poses[:FIXED_ROUTE_SIZE]
        else:
            last_point = route_poses[-1]
            padding = np.tile(last_point,
                              (FIXED_ROUTE_SIZE - len(route_poses), 1))
            route_poses = np.vstack([route_poses, padding])
        # Compose the route vectors for VectorNet.
        route_vec = np.zeros((route_poses.shape[0] - 1, 2 * ROUTE_POSE_SIZE),
                             np.float32)
        for idx in range(route_vec.shape[0]):
            route_vec[idx] = np.concatenate(
                (route_poses[idx], route_poses[idx + 1]), axis=0)
        return route_vec

    def _get_semantic_label(self, sample):
        semantic_label_lookup = {}
        for sid, label in sample['robot.front_camera.left.segmentation_info'][
                'idToLabels'].items():
            if label and 'class' in label:
                semantic_label_lookup[sid] = SIM_LABELS_MAPPING.get(
                    label['class'], SemanticLabel.BACKGROUND)
        semantic_labels = np.array([
            semantic_label_lookup.get(str(label), SemanticLabel.BACKGROUND)
            for label in sample['robot.front_camera.left.segmentation_image']
        ]).astype(np.uint8)
        semantic_labels_shape = sample[
            'robot.front_camera.left.segmentation_image.shape']
        semantic_labels = semantic_labels.reshape(semantic_labels_shape[0],
                                                  semantic_labels_shape[1], 1)
        return np.transpose(semantic_labels, (2, 0, 1))

    def _get_action(self, sample):
        action = np.zeros(6)
        action[0] = sample['robot.action'][0]
        action[5] = sample['robot.action'][1]
        return action

    def _get_speed(self, sample):
        speed = np.array([sample['robot.linear_velocity'][0]])
        return speed

    def _down_sample_input_image(self, batch):
        size = INPUT_IMAGE_SIZE
        batch['image'] = interpolate_resize(
            batch['image'],
            size,
            mode=tvf.InterpolationMode.BILINEAR,
        )
        if 'semantic_label' in batch:
            batch['semantic_label'] = interpolate_resize(
                batch['semantic_label'],
                size,
                mode=tvf.InterpolationMode.NEAREST)

    def _compose_semantic_labels(self, batch):
        batch['semantic_label_1'] = batch['semantic_label']
        h, w = batch['semantic_label_1'].shape[-2:]
        for downsample_factor in [2, 4]:
            size = h // downsample_factor, w // downsample_factor
            previous_label_factor = downsample_factor // 2
            batch[f'semantic_label_{downsample_factor}'] = interpolate_resize(
                batch[f'semantic_label_{previous_label_factor}'],
                size,
                mode=tvf.InterpolationMode.NEAREST)

    def _compose_rgb_labels(self, batch):
        batch['rgb_label_1'] = batch['image']
        h, w = batch['rgb_label_1'].shape[-2:]
        for downsample_factor in [2, 4]:
            size = h // downsample_factor, w // downsample_factor
            previous_label_factor = downsample_factor // 2
            batch[f'rgb_label_{downsample_factor}'] = interpolate_resize(
                batch[f'rgb_label_{previous_label_factor}'],
                size,
                mode=tvf.InterpolationMode.BILINEAR,
            )
