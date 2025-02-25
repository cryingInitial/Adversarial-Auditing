# Final Model Auditing of DP-SGD
This repository contains the source code for the paper Adversarial Sample-Based Approach for Tighter Privacy Auditing in Final Model-Only Scenarios at [NeurIPS Workshop 2024](https://arxiv.org/pdf/2412.01756). We refer to [Nearly Tight Black-Box Auditing of Differentially Private Machine Learning
](https://github.com/spalabucr/bb-audit-dpsgd) (see acknowledgement below).

## Install
Dependencies are managed by `conda/mamba`.  
Required dependencies can be installed using the command `conda env create -f env.yml` and then run `conda activate bb_audit_dpsgd`.  

## Training Models
To train cnn models trained on the MNIST dataset with for auditing. You can use the pretrain.sh
(More command line options can be found inside the `audit_model.py` file).


## Auditing
To audit a cnn model trained on the MNIST dataset with $\varepsilon=10.0$ using adversarial sample. You can run the following command: (You can use audit.sh)
```bash
$ CUDA_VISIBLE_DEVICES=0 python3 adversarial.py \
    --epochs 5 \
    --num_models 512 \
    --lr 0.01 \
    --eps 10.0 \
    --target blank \
    --num_trials 10
```

## Acknowledgement
We used code base from [Nearly Tight Black-Box Auditing of Differentially Private Machine Learning
](https://github.com/spalabucr/bb-audit-dpsgd). 

Thank you for their awesome works!
