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

REQUIRED_COLUMNS = [
    'driving_command', 'ego_speed', 'path', 'camera_image', 'route_poses'
]
SEMANTIC_LABELS_COLUMNS = [
    'semantic_labels', 'perspective_semantic_image_shape'
]
SEMANTIC_IMAGE_COLUMNS = [
    'perspective_semantic_image', 'perspective_semantic_image_labels',
    'perspective_semantic_image_shape'
]


def interpolate_resize(x, size, mode=tvf.InterpolationMode.NEAREST):
    '''Resize the tensor with interpolation
    '''
    x = tvf.resize(x, size, interpolation=mode, antialias=True)
    return x


@gin.configurable
class XMobilityIsaacSimDataModule(pl.LightningDataModule):
    '''Datamodule with dataset collected from Isaac Sim.
    '''
    def __init__(self,
                 dataset_path: str,
                 batch_size: int,
                 sequence_length: int,
                 num_workers: int,
                 enable_semantic: bool = False,
                 enable_rgb_stylegan: bool = False,
                 is_gwm_pretrain: bool = False,
                 precomputed_semantic_label: bool = True,
                 use_lazy_loading: bool = True):
        super().__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.enable_semantic = enable_semantic
        self.enable_rgb_stylegan = enable_rgb_stylegan
        self.is_gwm_pretrain = is_gwm_pretrain
        self.precomputed_semantic_label = precomputed_semantic_label
        self.use_lazy_loading = use_lazy_loading
        self.dataset_path = dataset_path
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        # Initialize datasets - data will be loaded on-demand during training
        # rather than all at once, thanks to the lazy-loading implementation in IsaacSimDataset
        if stage == 'fit':
            self.train_dataset = IsaacSimDataset(
                os.path.join(self.dataset_path, 'train'), self.sequence_length,
                self.enable_semantic, self.enable_rgb_stylegan,
                self.is_gwm_pretrain, self.precomputed_semantic_label,
                self.use_lazy_loading)
            self.val_dataset = IsaacSimDataset(
                os.path.join(self.dataset_path, 'val'), self.sequence_length,
                self.enable_semantic, self.enable_rgb_stylegan,
                self.is_gwm_pretrain, self.precomputed_semantic_label,
                self.use_lazy_loading)
        if stage == 'test' or stage is None:
            self.test_dataset = IsaacSimDataset(
                os.path.join(self.dataset_path, 'test'), self.sequence_length,
                self.enable_semantic, self.enable_rgb_stylegan,
                self.is_gwm_pretrain, self.precomputed_semantic_label,
                self.use_lazy_loading)

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
        test_dataset = IsaacSimDataset(os.path.join(self.dataset_path, 'test'),
                                       self.sequence_length,
                                       self.enable_semantic,
                                       self.enable_rgb_stylegan,
                                       self.is_gwm_pretrain,
                                       self.use_lazy_loading)
        return DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )


class IsaacSimDataset(Dataset):
    '''Dataset from Isaac sim.
    '''
    def __init__(self,
                 dataset_path: str,
                 sequence_length: int,
                 enable_semantic: bool = False,
                 enable_rgb_stylegan: bool = False,
                 is_gwm_pretrain: bool = False,
                 precomputed_semantic_label: bool = True,
                 use_lazy_loading: bool = False):
        super().__init__()
        self.sequence_length = sequence_length
        self.enable_semantic = enable_semantic
        self.enable_rgb_stylegan = enable_rgb_stylegan
        self.is_gwm_pretrain = is_gwm_pretrain
        self.precomputed_semantic_label = precomputed_semantic_label
        self.use_lazy_loading = use_lazy_loading
        
        # Get the required columns to load data.
        self.required_columns = REQUIRED_COLUMNS
        if self.enable_semantic:
            if precomputed_semantic_label:
                self.required_columns = REQUIRED_COLUMNS + SEMANTIC_LABELS_COLUMNS
            else:
                self.required_columns = REQUIRED_COLUMNS + SEMANTIC_IMAGE_COLUMNS
        
        if self.use_lazy_loading:
            # Lazy loading implementation
            self.file_paths = []  # Store file paths instead of loading data
            self.file_sizes = []  # Store the number of rows in each file
            self.accumulated_sample_sizes = [0]  # Start with 0
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
                          desc=f"Indexing data from {scenario_path}",
                          unit="file") as pbar:
                    for run_file in run_files:
                        parquet_path = os.path.join(scenario_path, run_file)
                        # Just get the number of rows without loading the entire file
                        sample_count = self._get_sample_count(parquet_path)
                        self.file_paths.append(parquet_path)
                        self.file_sizes.append(sample_count)
                        
                        # Calculate how many complete sequences we can get from this file
                        usable_samples = sample_count // self.sequence_length
                        self.num_samples += usable_samples
                        self.accumulated_sample_sizes.append(self.num_samples)
                        pbar.update(1)
        else:
            # Original implementation - load everything into memory
            self.dfs = []
            self.accumulated_sample_sizes = [0]
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
                        df = pd.read_parquet(parquet_path, columns=self.required_columns, engine='pyarrow')
                        self.dfs.append(df)
                        self.num_samples += len(df) // self.sequence_length
                        self.accumulated_sample_sizes.append(self.num_samples)
                        pbar.update(1)

    def _get_sample_count(self, parquet_path):
        # Read only the metadata to get the number of rows
        import pyarrow.parquet as pq
        metadata = pq.read_metadata(parquet_path)
        return metadata.num_rows

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        batch = {}
        
        if self.use_lazy_loading:
            # Lazy loading implementation
            # Find which file contains this index
            file_idx = bisect.bisect_right(self.accumulated_sample_sizes, index) - 1
            file_path = self.file_paths[file_idx]
            
            # Calculate the sequence start position within the file
            relative_index = index - self.accumulated_sample_sizes[file_idx]
            sequence_start = relative_index * self.sequence_length
            
            # Load the entire sequence at once
            df = pd.read_parquet(
                file_path,
                columns=self.required_columns,
                engine='pyarrow'
            )
            
            # Extract just the rows we need
            sequence_end = sequence_start + self.sequence_length
            sequence_df = df.iloc[sequence_start:sequence_end]
            
            # Process the loaded data
            for seq_idx in range(self.sequence_length):
                element = self._get_element(sequence_df, seq_idx)
                for k, v in element.items():
                    batch[k] = batch.get(k, []) + [v]
        else:
            # Original implementation
            df_idx = bisect.bisect_left(self.accumulated_sample_sizes, index + 1) - 1
            sample_idx = (index - self.accumulated_sample_sizes[df_idx]) * self.sequence_length
            
            for seq_idx in range(self.sequence_length):
                element = self._get_element(self.dfs[df_idx], sample_idx + seq_idx)
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
            element['path'] = self._get_path(sample)
            element['route_vectors'] = self._get_route_vector(sample)
        return element

    def _get_rgb_image(self, sample):
        rgb_image = Image.open(io.BytesIO(sample['camera_image']))
        return np.transpose(np.array(rgb_image), (2, 0, 1)) / 255.0

    def _get_route_vector(self, sample):
        route_poses = np.array(sample['route_poses'], np.float32)
        route_poses = route_poses.reshape(
            len(route_poses) // ROUTE_POSE_SIZE, ROUTE_POSE_SIZE)

        route_vec = np.zeros((route_poses.shape[0] - 1, 2 * ROUTE_POSE_SIZE),
                             np.float32)
        for idx in range(route_vec.shape[0]):
            route_vec[idx] = np.concatenate(
                (route_poses[idx], route_poses[idx + 1]), axis=0)

        return route_vec

    def _get_semantic_label(self, sample):
        if self.precomputed_semantic_label:
            semantic_labels = np.array(sample['semantic_labels'],
                                       dtype=np.uint8)
        else:
            semantic_label_lookup = {}
            for sid, label in sample[
                    'perspective_semantic_image_labels'].items():
                if label and 'class' in label:
                    semantic_label_lookup[sid] = SIM_LABELS_MAPPING.get(
                        label['class'], SemanticLabel.BACKGROUND)
            semantic_labels = np.array([
                semantic_label_lookup.get(str(label), SemanticLabel.BACKGROUND)
                for label in sample['perspective_semantic_image']
            ]).astype(np.uint8)
        semantic_labels = semantic_labels.reshape(
            sample['perspective_semantic_image_shape'])
        return np.transpose(semantic_labels, (2, 0, 1))

    def _get_action(self, sample):
        return sample['driving_command']

    def _get_speed(self, sample):
        speed = np.zeros(1)
        speed[0] = sample['ego_speed']
        return speed

    def _get_path(self, sample):
        return np.array(sample['path'], np.float32)

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
