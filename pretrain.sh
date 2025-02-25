SEEDS=(0)
EPS=(10.0)
GPUS=(0)
N_EPOCHS=100
TARGET_TYPE="blank"
SEP="_"

if [ ${#SEEDS[@]} -ne ${#GPUS[@]} ]; then
    echo "SEEDS and GPUS must have the same length"
    exit 1
fi

for eps_idx in ${!EPS[@]}
do
    for seed_idx in ${!SEEDS[@]}
    do
        mkdir -p exp_data/mnist_$TARGET_TYPE$SEP$N_EPOCHS/seed${SEEDS[$seed_idx]}/
        CUDA_VISIBLE_DEVICES=${GPUS[$seed_idx]} python3 pretrain.py --data_name mnist --model_name cnn --n_epochs $N_EPOCHS --lr 6.67e-5 --epsilon ${EPS[$eps_idx]} \
            --target_type $TARGET_TYPE --n_reps 2048 \
            --seed ${SEEDS[$seed_idx]} --out exp_data/mnist_$TARGET_TYPE$SEP$N_EPOCHS/seed${SEEDS[$seed_idx]}/ --block_size 10000 > exp_data/mnist_$TARGET_TYPE$SEP$N_EPOCHS/seed${SEEDS[$seed_idx]}/${EPS[$eps_idx]}.log 2>&1 &
    done
done
