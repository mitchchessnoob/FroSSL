

EXPERIMENT_NAME="eurosat_ivne_finetune"
DATASET="eurosat_msi"
BACKBONE="resnet50" 
CONFIG_NAME="ivne"

echo "run $CONFIG_NAME."

# python3 -u main_pretrain.py \
#     --config-path "scripts/pretrain/eurosat_msi" \
#     --config-name "$CONFIG_NAME" \
#     ++name="$EXPERIMENT_NAME-pretrain" \
#     ++backbone.name="$BACKBONE"  \
#     ++data.dataset="$DATASET" 

echo "trained_models/ivne/w6cymg92/eurosat_ivne-pretrain-w6cymg92-ep=399.ckpt" > last_ckpt.txt

# get pretrained path from last_ckpt.txt file
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