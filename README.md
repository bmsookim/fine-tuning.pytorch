# Pytorch-mammo
Pytorch implementation of Digital Mammography Classification

## Requirements
See the [installation instruction](INSTALL.md) for a step-by-step installation guide.
See the [server instruction](SERVER.md) for server settup.
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downloads)
- Install [cudnn v5.1](https://developer.nvidia.com/cudnn)
- Download [Pytorch 2.7](https://pytorch.org) and clone the repository.
```bash
pip install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp27-none-linux_x86_64.whl
pip install torchvision
git clone https://github.com/meliketoy/wide-resnet.pytorch
```

## How to run
After you have cloned the repository, you can train the dataset by running the script below.
```bash
# zero-base training
python main --lr [:lr]

# fine-tuning
python main --finetune --lr [:lr]

# fine-tuning with additional linear layers
python main --finetune --addlayer --lr [:lr]
```
