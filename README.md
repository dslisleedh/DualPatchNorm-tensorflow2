# DualPatchNorm-tensorflow2
Unofficial implementation of DualPatchNorm using tensorflow2. Include ablation code too.  
Paper: [Dual PatchNorm(Google Research and Brain Team)](https://arxiv.org/abs/2302.01327)

## How to run  
    git clone git@github.com:dslisleedh/DualPatchNorm-tensorflow2.git
    cd DualPatchNorm-tensorflow2
    conda env create -f environment.yaml
    conda activate dpn
    train.py 

Default setting is run hpo that takes lots of time. 
Disable HPO if you just want to run code for once. 

    train.py hpo=False hparams.$ANY_HPARAMS_IN_CONFIG=...

