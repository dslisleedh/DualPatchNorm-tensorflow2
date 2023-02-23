# DualPatchNorm-tensorflow2
Unofficial implementation of DualPatchNorm using tensorflow2. Include ablation code too.  
Paper: [Dual PatchNorm(Google Research and Brain Team)](https://arxiv.org/abs/2302.01327)

## Note  
Not completed codes.

## How to run  
    git clone git@github.com:dslisleedh/DualPatchNorm-tensorflow2.git
    cd DualPatchNorm-tensorflow2
    conda env create -f environment.yaml
    conda activate dpn
    python train.py 

If you have never downloaded the imagenet_resized 64x64 using tensorflow_datasets, run `donwload_datasets.py` before run `train.py`  
Default config is set run HPO that takes lots of time.  
Disable HPO if you just want to run code for once.  

    python train.py hpo=False hparams.$ANY_HPARAMS_IN_CONFIG=...

