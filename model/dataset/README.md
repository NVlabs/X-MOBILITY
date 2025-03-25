# Dataset

This directory contains the dataset loaders used for training and evaluating the X-MOBILITY models.

## Dataset Classes

### IsaacSimDataset
Dataloader for dataset collected with [Isaac Sim Replicator](https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/index.html) and Nav2 navigation stack. Example available at https://huggingface.co/datasets/nvidia/X-Mobility.

### MobilityGenDataset
Dataloader for dataset collected with [MobilityGen](https://github.com/NVLabs/MobilityGen) pipeline.

## Data Format
Each dataset is stored in Parquet (`.pqt`) files, which provide efficient columnar storage for structured data.

## Data Folder Structure
The dataset is organized into train, validation, and test splits, with multiple scenarios in each:

```
data
 - train
   - scenario_0
      - run_0000.pqt
        run_0001.pqt
        ...
   - scenario_1
      - run_0000.pqt
        run_0001.pqt
        ...
 - val
    - scenario_0
      - run_0000.pqt
        run_0001.pqt
        ...
    - scenario_1
      - run_0000.pqt
        run_0001.pqt
        ...
 - test
    - scenario_0
      - run_0000.pqt
        run_0001.pqt
        ...
    - scenario_1
      - run_0000.pqt
        run_0001.pqt
        ...