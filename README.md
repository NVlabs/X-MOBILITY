<h1 align="center">X-Mobility: E2E Generalizable Navigation with World Modeling</h1>
</p>

This is the PyTorch implementation for training and deployment of <a href="https://arxiv.org/abs/2410.17491">X-Mobility</a>.

<p align="center">
    <img src="images/x_mobility.png" alt="X-Mobility" width="600" >
    <br/> X-Mobility: A generalizable navigation model using auto-regressive world modeling, multi-head decoders, and decoupled policy learning for robust, zero-shot Sim2Real and cross-embodiment transfer.
</p>



## Setup
1. Install docker

2. Build the docker image
```
docker build --network=host -t <image-name> .
```

3. Download the datasets
[TBD]

## Usages
1. Launch the docker image and set WANDB_API_KEY inside:
```
docker run --shm-size=64g -v <path-to-datasets>:/workspace/datasets -it <image-name> bash
export WANDB_API_KEY=<wandb-api-key>
```

2. Training with action policy enabled:
```
python3 train.py -c configs/train_config.gin -d datasets/ -o <output-dir>
```

3. Training with world model only:
```
python3 train.py -c configs/pretrained_gwm_train_config.gin -d datasets/ -o <output-dir>
```

4. Evaluating checkpoint
```
python3 evaluate.py -c configs/train_config.gin -d datasets/ -p <checkpoint>
```

5. ONNX & TensorRT conversion
```
python3 onnx_conversion.py -p <checkpoint> -o <onnx_file_path>

python3 trt_conversion.py -o <onnx_file_path> -t <trt_file_path>
```

6. E2E navigation with ROS2
With the TensorRT engine, follow this [instruction](./ros2_deployment/README.md) to setup ROS2, Isaac Sim to run a demo example of X-Mobility.

## License
X-Mobility is released under the Apache License 2.0. See LICENSE for additional details.

## Core Contributors
Wei Liu, Huihua Zhao, Chenran Li, Joydeep Biswas, Billy Okal, Pulkit Goyal, Yan Chang, and Soha Pouya


## Acknowledgments
We would like to acknowledge the following projects where parts of the codes in this repo is derived from:
- [MILE](https://github.com/wayveai/mile)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [Diffusers](https://github.com/huggingface/diffusers)
- [DINOv2](https://github.com/facebookresearch/dinov2)

## Citing X-Mobility:

If you find this repository useful, please consider citing:
```bibtex
@article{liu2024x,
  title={X-mobility: End-to-end generalizable navigation via world modeling},
  author={Liu, Wei and Zhao, Huihua and Li, Chenran and Biswas, Joydeep and Okal, Billy and Goyal, Pulkit and Chang, Yan and Pouya, Soha},
  journal={arXiv preprint arXiv:2410.17491},
  year={2024}
}
```
