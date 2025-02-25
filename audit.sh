CUDA_VISIBLE_DEVICES=0 python3 adversarial.py \
    --epochs 5 \
    --num_models 8 \
    --lr 0.01 \
    --eps 10.0 \
    --target blank \
    --num_trials 10
