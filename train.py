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

import os

import gin
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from arg_parser import parse_arguments, TaskMode
from model.dataset.lerobot_dataset import XMobilityLeRobotDataModule  # pylint: disable=unused-import
from model.dataset.mobility_gen_dataset import XMobilityMobilityGenDataModule  # pylint: disable=unused-import
from model.trainer import XMobilityTrainer  # pylint: disable=unused-import


@gin.configurable
def train(dataset_path, output_dir, ckpt_path, wandb_entity_name,
          wandb_project_name, wandb_run_name, precision, epochs, data_module,
          model_trainer):
    # Create a output directory if not exit.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = data_module(dataset_path=dataset_path)
    if ckpt_path:
        model = model_trainer.load_from_checkpoint(checkpoint_path=ckpt_path,
                                                   strict=False)
    else:
        model = model_trainer()

    wandb_logger = WandbLogger(entity=wandb_entity_name,
                               project=wandb_project_name,
                               name=wandb_run_name,
                               save_dir=output_dir,
                               group="DDP",
                               log_model=True)

    callbacks = [
        pl.callbacks.ModelSummary(-1),
        pl.callbacks.LearningRateMonitor(),
        ModelCheckpoint(dirpath=os.path.join(output_dir, 'checkpoints'),
                        save_top_k=3,
                        monitor='val_loss',
                        mode='min',
                        save_last=True),
    ]
    trainer = pl.Trainer(max_epochs=epochs,
                         precision=precision,
                         sync_batchnorm=True,
                         callbacks=callbacks,
                         strategy='ddp_find_unused_parameters_true',
                         logger=wandb_logger)
    trainer.fit(model, datamodule=data)

    trainer.test(ckpt_path="last", datamodule=data)

    return wandb_logger


def log_gin_config(logger: WandbLogger):
    # This function should be called after all the gin configurable functions.
    # Otherwise, the config string will be empty.
    gin_config_str = gin.operative_config_str()

    # Create a temporary file to store the gin config
    with open("/tmp/gin_config.txt", "w", encoding='UTF-8') as f:
        f.write(gin_config_str)

    # Log the artifact using the WandbLogger
    artifact = wandb.Artifact("gin_config", type="text")
    artifact.add_file("/tmp/gin_config.txt")
    logger.experiment.log_artifact(artifact)


def main():
    args = parse_arguments(TaskMode.TRAIN)

    for config_file in args.config_files:
        gin.parse_config_file(config_file, skip_unknown=True)

    # Run the training loop.
    wandb_logger = train(args.dataset_path, args.output_dir,
                         args.checkpoint_path, args.wandb_entity_name,
                         args.wandb_project_name, args.wandb_run_name)

    # Log gin config
    log_gin_config(wandb_logger)

    # Finish wandb
    wandb.finish()


if __name__ == '__main__':
    main()
