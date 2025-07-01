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

import numpy as np
import torch
from torchmetrics.functional import jaccard_index
from tqdm import tqdm
from PIL import Image

from model.dataset.isaac_sim_semantic_label import SemanticLabel
from model.dataset.lerobot_semantic_label import LeRobotSemanticLabel
from model.visualization import SEMANTIC_COLORS
from model.x_mobility.utils import pack_sequence_dim, unpack_sequence_dim


def calculate_iou_metric(batch, output, semantic_label_names):
    ''' Log sementic segmentation IOU (intersection over union) metric scores.

     Returns:
        Calculated IOU metric
     '''
    return jaccard_index(
        preds=output,
        target=batch,
        task='multiclass',
        num_classes=len(semantic_label_names),
        average='none',
    )


class PredictionEvaulator:
    '''Evaluator for WM prediction
    '''
    def __init__(self,
                 model,
                 datamodle,
                 wandb_logger,
                 max_history_length=5,
                 max_future_length=[1, 5, 10, 15, 20],
                 use_trained_policy=True):
        self.model = model
        self.dataloader = datamodle.load_test_data()
        self.logger = wandb_logger

        self.use_trained_policy = use_trained_policy
        self.max_history_length = max_history_length
        self.max_future_length = max_future_length

        self.evaluation_result = {}
        for length in max_future_length:
            self.evaluation_result[
                f'history_{self.max_history_length}_horizon_{length}'] = {}
        for i in range(self.max_history_length):
            self.evaluation_result[f'history_{i}_one_step_prediction'] = {}
            self.evaluation_result[f'history_{i}_one_step_reconstruction'] = {}

    def compute(self):
        batch_id = 0
        total_batches = self.dataloader.dataset.num_samples
        with tqdm(total=total_batches,
                  desc=f"Evaluating batch {batch_id}",
                  unit="batch") as pbar:
            for batch in self.dataloader:
                batch = {
                    key: value.to(self.model.device)
                    for key, value in batch.items()
                }
                evaluation_result = {}
                b, _ = batch['image'].shape[:2]

                infer_batch = {}
                infer_batch['history'] = batch['speed'].new_zeros(
                    (b, self.model.model.rssm.hidden_state_dim)).float()
                infer_batch['sample'] = batch['speed'].new_zeros(
                    (b, self.model.model.rssm.state_dim)).float()
                infer_batch['action'] = torch.zeros_like(batch['action'][:, 0])

                # Observation steps
                for i in range(self.max_history_length):
                    infer_batch['image'] = batch['image'][:, i:i + 1]
                    infer_batch['speed'] = batch['speed'][:, i:i + 1]
                    history_pre, sample_pre, _, _ = self.model.inference_prediction(
                        infer_batch, False, False)
                    _, _, history, sample, _, _, _ = self.model.inference(
                        infer_batch, False, False, False)

                    state_pre = torch.cat([history_pre, sample_pre], dim=-1)
                    state = torch.cat([history, sample], dim=-1)

                    evaluation_result[f'history_{i}_one_step_prediction'] = {}
                    evaluation_result[
                        f'history_{i}_one_step_reconstruction'] = {}

                    if self.model.model.enable_semantic:
                        target = batch['semantic_label'][:, i, 0]

                        reconstruction_iou = self.semantic_eval(state, target)
                        prediction_iou = self.semantic_eval(state_pre, target)

                        evaluation_result[f'history_{i}_one_step_prediction'][
                            "Semantic"] = prediction_iou
                        evaluation_result[
                            f'history_{i}_one_step_reconstruction'][
                                "Semantic"] = reconstruction_iou

                    infer_batch['history'] = history
                    infer_batch['sample'] = sample
                    infer_batch['action'] = batch['action'][:, i]

                # Prediction steps
                for i in range(np.max(self.max_future_length)):
                    history_pre, sample_pre, _, _ = self.model.inference_prediction(
                        infer_batch, False, False)
                    state_pre = torch.cat([history_pre, sample_pre], dim=-1)
                    infer_batch['history'] = history_pre
                    infer_batch['sample'] = sample_pre
                    # Get action.
                    if self.use_trained_policy:
                        infer_batch['route'] = batch[
                            'route_vectors'][:, i + self.max_history_length:i +
                                             self.max_history_length + 1]
                        action_commands, _ = self.model.model.action_policy.inference(
                            state_pre, infer_batch)
                        infer_batch['action'] = pack_sequence_dim(
                            action_commands)
                    else:
                        infer_batch['action'] = batch[
                            'action'][:, i + self.max_history_length]

                    all_state_pre = unpack_sequence_dim(
                        state_pre, b, 1) if i == 0 else torch.cat([
                            all_state_pre,
                            unpack_sequence_dim(state_pre, b, 1)
                        ],
                                                                  dim=1)

                    if i + 1 in self.max_future_length:
                        evaluation_result[
                            f'history_{self.max_history_length}_horizon_{i+1}'] = {
                                "FE": {},
                                "AE": {}
                            }
                        trajectory = pack_sequence_dim(all_state_pre)
                        if self.model.model.enable_semantic:
                            target = batch[
                                'semantic_label'][:,
                                                  i + self.max_history_length,
                                                  0]
                            prediction_iou = self.semantic_eval(
                                state_pre,
                                target,
                                viz_config=(
                                    False, batch_id, i,
                                    batch['image'][:, i +
                                                   self.max_history_length]))
                            evaluation_result[
                                f'history_{self.max_history_length}_horizon_{i+1}'][
                                    "FE"]["Semantic"] = prediction_iou

                            target = batch[
                                'semantic_label'][:,
                                                  self.max_history_length:i +
                                                  1 + self.max_history_length,
                                                  0]
                            target = pack_sequence_dim(target.contiguous())
                            prediction_iou = self.semantic_eval(
                                trajectory, target)
                            evaluation_result[
                                f'history_{self.max_history_length}_horizon_{i+1}'][
                                    "AE"]["Semantic"] = prediction_iou

                self.log(evaluation_result)
                batch_id = batch_id + 1
                pbar.update(1)
        self.print_result_mean()
        self.log_mean()

    def semantic_eval(self, state, target, viz_config=(False, 0, 0, None)):
        target = target.cpu().detach()
        semantic_result = self.model.model.semantic_decoder(
            state)['semantic_segmentation_1'].cpu().detach()

        semantic_result = torch.argmax(semantic_result.detach(), dim=-3)
        semantic_label_names = SemanticLabel.get_semantic_lable_names()
        iou_scores = calculate_iou_metric(target, semantic_result,
                                          semantic_label_names)
        iou_dict = {}
        for semantic_class_name, iou_score in zip(semantic_label_names,
                                                  iou_scores):
            iou_dict[semantic_class_name] = iou_score

        enable_viz, batch_id, step, rgb_gt = viz_config
        if enable_viz:
            color_map = torch.tensor(SEMANTIC_COLORS,
                                     dtype=torch.uint8,
                                     device=target.device)
            # Broadcast to shape [b, h, w, 3]
            target = color_map[target.int()]
            semantic_result = color_map[semantic_result.int()]

            # Rearrange the rgb channel [b, 3, h, w]
            target = target.permute(0, 3, 1, 2)
            semantic_result = semantic_result.permute(0, 3, 1, 2)

            semantic_image = semantic_result[0].permute(1, 2, 0).cpu().numpy()
            Image.fromarray(semantic_image).save(
                f"/tmp/wm_prediction_logged/batch_{batch_id}_semantic_step_{step}.png"
            )

            rgb_viz_image = (
                np.clip(
                    rgb_gt[0].permute(1, 2, 0).cpu().numpy() * 255,  # pylint: disable=unsubscriptable-object
                    0,
                    255)).astype(np.uint8)
            Image.fromarray(rgb_viz_image).save(
                f"/tmp/wm_prediction_logged/batch_{batch_id}_rgb_step_{step}.png"
            )

        return iou_dict

    def log(self, evaluation_result):
        for setting_key in self.evaluation_result.keys():  # pylint: disable=consider-iterating-dictionary, consider-using-dict-items
            evaluation_setting = evaluation_result[setting_key]

            for item_key, evaluation_item in evaluation_setting.items():
                if item_key in ["FE", "AE"]:
                    for inner_item_key, evaluation_inner_item in evaluation_item.items(
                    ):
                        for metric_key, metric_vlaue in evaluation_inner_item.items(
                        ):
                            current_name = item_key + "/" + inner_item_key + "/" + metric_key
                            if current_name not in self.evaluation_result[
                                    setting_key]:
                                self.evaluation_result[setting_key][
                                    current_name] = metric_vlaue.reshape(
                                        1, -1).numpy()
                            else:
                                self.evaluation_result[setting_key][
                                    current_name] = np.concatenate(
                                        (self.evaluation_result[setting_key]
                                         [current_name],
                                         metric_vlaue.reshape(1, -1).numpy()),
                                        axis=0)
                else:
                    for metric_key, metric_vlaue in evaluation_item.items():
                        current_name = item_key + "/" + metric_key
                        if current_name not in self.evaluation_result[
                                setting_key]:
                            self.evaluation_result[setting_key][
                                current_name] = metric_vlaue.reshape(
                                    1, -1).numpy()
                        else:
                            self.evaluation_result[setting_key][
                                current_name] = np.concatenate(
                                    (self.evaluation_result[setting_key]
                                     [current_name], metric_vlaue.reshape(
                                         1, -1).numpy()),
                                    axis=0)

    def print_result_all(self):
        print("________________________")
        for setting_key, evaluation_setting in self.evaluation_result.items():
            print("*********")
            print(setting_key + ":")
            for item_key in evaluation_setting.keys():
                print(item_key + ":", evaluation_setting[item_key])

    def print_result_mean(self):
        print("________________________")
        for setting_key, evaluation_setting in self.evaluation_result.items():
            print("*********")
            print(setting_key + ":")
            for item_key in evaluation_setting.keys():
                print(item_key + ":",
                      (evaluation_setting[item_key]).mean(axis=0), "+/-",
                      (evaluation_setting[item_key]).std(axis=0))

    def log_mean(self):
        log_dict = {}
        for setting_key, evaluation_setting in self.evaluation_result.items():
            for item_key in evaluation_setting.keys():
                mean = np.mean(evaluation_setting[item_key], axis=0).tolist()
                std = np.std(evaluation_setting[item_key], axis=0).tolist()
                if len(mean) > 1:
                    for i in range(len(mean)):
                        log_dict[f'{setting_key}/{item_key}_{i}_mean'] = mean[
                            i]
                        log_dict[f'{setting_key}/{item_key}_{i}_std'] = std[i]
                else:
                    log_dict[f'{setting_key}/{item_key}_mean'] = mean[0]
                    log_dict[f'{setting_key}/{item_key}_std'] = std[0]
        self.logger.experiment.log(log_dict, step=0)
