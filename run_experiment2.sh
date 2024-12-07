
EXPERIMENT_NAME="mit67_initial"
DATASET="mit67"
BACKBONE="resnet50" 
CONFIG_NAME="frossl_sym"

echo "run $CONFIG_NAME experiment with $DATASET."


python3 -u main_pretrain.py \
    --config-path "scripts/pretrain/mit67" \
    --config-name "$CONFIG_NAME" \
    ++name="$EXPERIMENT_NAME-pretrain" \
    ++backbone.name="$BACKBONE"  \
    ++data.dataset="$DATASET" 

get pretrained path from last_ckpt.txt file
TRAINED_CHECKPOINT_PATH=$(cat last_ckpt.txt)
TRAINED_CHECKPOINT_WANDB_ID=$(echo $TRAINED_CHECKPOINT_PATH | awk -F '/' '{print $3}')
echo "$TRAINED_CHECKPOINT_WANDB_ID $TRAINED_CHECKPOINT_PATH"


python3 -u main_linear.py \
    --config-path "scripts/linear/$DATASET"  \
    --config-name "$CONFIG_NAME" \
    ++data.dataset="$DATASET" \
    ++name="$EXPERIMENT_NAME-linear-$TRAINED_CHECKPOINT_WANDB_ID" \

python3 -c "from main_test import main; main('scripts/linear/$DATASET/$CONFIG_NAME.yaml')"

echo "done"