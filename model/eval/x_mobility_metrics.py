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

import torch
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics.functional import jaccard_index, mean_absolute_error, mean_absolute_percentage_error

from model.dataset.lerobot_semantic_label import LeRobotSemanticLabel
from model.dataset.isaac_sim_dataset import SemanticLabel


class XMobilityMetrics():
    '''Metrics calculator for x-mobility model.
    '''
    def evaluate(self, batch, output, prefix):
        metrics = {}
        metrics.update(self._semantic_iou(batch, output, prefix))
        metrics.update(self._semantic_f1_score(batch, output, prefix))
        metrics.update(self._action_errors(batch, output, prefix))
        metrics.update(self._path_errors(batch, output, prefix))
        return metrics

    def _semantic_iou(self, batch, output, prefix):
        if 'semantic_segmentation_1' not in output or 'semantic_label' not in batch:
            return {}
        # Sementic segmentation IOU (intersection over union) metric scores.
        metrics = {}
        semantic_label_names = SemanticLabel.get_semantic_lable_names()
        pred = torch.argmax(output['semantic_segmentation_1'].detach(), dim=-3)
        target = batch['semantic_label'].reshape_as(pred).detach()  # ensure target is the same size as pred
        iou_scores = jaccard_index(preds=pred,
                                   target=target,
                                   task='multiclass',
                                   num_classes=len(semantic_label_names),
                                   average='none')
        for semantic_class_name, iou_score in zip(semantic_label_names,
                                                  iou_scores):
            metrics[
                f'{prefix}/metrics/semantic/iou_{semantic_class_name}'] = iou_score
        return metrics

    def _semantic_f1_score(self, batch, output, prefix):
        # Sementic segmentation F1 metric scores.
        if 'semantic_segmentation_1' not in output or 'semantic_label' not in batch:
            return {}

        # Some batches have no instances of the classes, might cause warnings printed out.
        if prefix != 'test':
            return {}

        semantic_label_names = SemanticLabel.get_semantic_lable_names()
        target = batch['semantic_label'].detach().view(-1).long()
        pred = torch.argmax(output['semantic_segmentation_1'],
                            dim=-3).detach().view(-1).long()
        f1_scores = multiclass_f1_score(input=pred,
                                        target=target,
                                        num_classes=len(semantic_label_names),
                                        average=None)
        metrics = {}
        for semantic_class_name, f1_score in zip(semantic_label_names,
                                                 f1_scores):
            metrics[
                f'{prefix}/metrics/semantic/f1_{semantic_class_name}'] = f1_score
        return metrics

    def _action_errors(self, batch, output, prefix):
        if 'action' not in output or 'action' not in batch:
            return {}
        # Metrics for action prediction.
        target_action = batch['action'].detach()
        pred_action = output['action'].detach()

        # Now we are only using linear x and angular z for our robot.
        linear_x_idx = 0
        angular_z_idx = -1
        assert linear_x_idx < target_action.shape[-1]
        assert angular_z_idx < target_action.shape[-1]

        metrics = {}
        # Percentage error.
        linear_percentage_error = mean_absolute_percentage_error(
            pred_action[:, :, linear_x_idx], target_action[:, :, linear_x_idx])
        angular_percentage_error = mean_absolute_percentage_error(
            pred_action[:, :, angular_z_idx], target_action[:, :,
                                                            angular_z_idx])

        metrics[
            f'{prefix}/metrics/action/linear_percentage_error'] = linear_percentage_error
        metrics[
            f'{prefix}/metrics/action/angular_percentage_error'] = angular_percentage_error

        # MSE metrics.
        linear_mae = mean_absolute_error(pred_action[:, :, linear_x_idx],
                                         target_action[:, :, linear_x_idx])
        angular_mae = mean_absolute_error(pred_action[:, :, angular_z_idx],
                                          target_action[:, :, angular_z_idx])
        metrics[f'{prefix}/metrics/action/linear_mae'] = linear_mae
        metrics[f'{prefix}/metrics/action/angular_mae'] = angular_mae

        return metrics

    def _path_errors(self, batch, output, prefix):
        if 'path' not in output or 'path' not in batch:
            return {}
        # Metrics for path prediction.
        target_path = batch['path'].detach()
        pred_path = output['path'].detach()

        metrics = {}
        # MSE metrics.
        metrics[f'{prefix}/metrics/action/path_mae'] = mean_absolute_error(
            pred_path, target_path)

        return metrics
