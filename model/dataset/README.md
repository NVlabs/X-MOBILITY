# Dataset

This directory contains the dataset loaders used for training and evaluating the X-MOBILITY models.

## IsaacSimDataset
Dataloader for dataset collected with [Isaac Sim Replicator](https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/index.html) and Nav2 navigation stack. Example available at https://huggingface.co/datasets/nvidia/X-Mobility.

### Data Format
Each dataset is stored in Parquet (`.pqt`) files, which provide efficient columnar storage for structured data.

### Data Folder Structure
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
```

## LeRobotDataset
Dataloader for dataset collected with [MobilityGen](https://github.com/NVLabs/MobilityGen) and exported to Lerobot.

### Data Format
Contained within a single folder, the LeRobotDataset format makes use of several ways to serialize data which can be 
useful to understand if you plan to work more closely with this format. Please refer to the 
[source repository](https://github.com/huggingface/lerobot/tree/main) for more information.

### Additional Considerations
Currently the XMobilityLeRobotDatasetModule requires the following changes to source in order to be used with XMobility:
 - The semantic images need to be 3 channels, due to the specification, but only a single channel is used; therefore, all channels need to be the same value
 - All references to SemanticLabel need to be replaced with LeRobotSemanticLabel or the equivalent for your dataset
 - All references to SEMANTIC_COLORS need to be replaced with LEROBOT_SEMANTIC_COLORS or the equivalent for your dataset
 - Either modify or add a gin config file such that it is representative of your dataset
 - FIXED_ROUTE_SIZE and ROUTE_POSE_SIZE in lerobot_dataset.py may need to be updated to reflect your dataset
 - The lerobot dataset's metadata (meta/info.json) will need to be manually modified to include your intended train, test, val split, see below; the values represent episodes.

```
"splits": {
    "train": "0:70",
    "validate" : "70:78",
    "test" : "78:86"
},
```
