echo "run comparism."

EXPERIMENT_NAME="eurosat_cropsize64"
DATASET="eurosat_msi"
BACKBONE="resnet18" 
CONFIG_NAME="frossl_hat"


python3 -u main_pretrain.py \
    --config-path "scripts/pretrain/eurosat_msi" \
    --config-name "$CONFIG_NAME" \
    ++name="$EXPERIMENT_NAME-pretrain" \
    ++backbone.name="$BACKBONE"  \
    ++data.dataset="$DATASET" 

# get pretrained path from last_ckpt.txt file
TRAINED_CHECKPOINT_PATH=$(cat last_ckpt.txt)
TRAINED_CHECKPOINT_WANDB_ID=$(echo $TRAINED_CHECKPOINT_PATH | awk -F '/' '{print $3}')
echo "$TRAINED_CHECKPOINT_WANDB_ID $TRAINED_CHECKPOINT_PATH"


python3 -u main_linear.py \
    --config-path "scripts/linear/$DATASET"  \
    --config-name "$CONFIG_NAME" \
    ++data.dataset="$DATASET" \
    ++name="$EXPERIMENT_NAME-linear-$TRAINED_CHECKPOINT_WANDB_ID" \



EXPERIMENT_NAME="eurosat_cropsize64_filter"
CONFIG_NAME="frossl"

python3 -u main_pretrain.py \
    --config-path "scripts/pretrain/eurosat_msi" \
    --config-name "$CONFIG_NAME" \
    ++name="$EXPERIMENT_NAME-pretrain" \
    ++backbone.name="$BACKBONE"  \
    ++data.dataset="$DATASET" 

# get pretrained path from last_ckpt.txt file
TRAINED_CHECKPOINT_PATH=$(cat last_ckpt.txt)
TRAINED_CHECKPOINT_WANDB_ID=$(echo $TRAINED_CHECKPOINT_PATH | awk -F '/' '{print $3}')
echo "$TRAINED_CHECKPOINT_WANDB_ID $TRAINED_CHECKPOINT_PATH"


python3 -u main_linear.py \
    --config-path "scripts/linear/$DATASET"  \
    --config-name "$CONFIG_NAME" \
    ++data.dataset="$DATASET" \
    ++name="$EXPERIMENT_NAME-linear-$TRAINED_CHECKPOINT_WANDB_ID" \

echo "done"