
EXPERIMENT_NAME="eurosat_final_model"
DATASET="eurosat_msi"
BACKBONE="resnet50" 
CONFIG_NAME="frossl"

echo "run $CONFIG_NAME."

python3 -u main_pretrain.py \
    --config-path "scripts/pretrain/eurosat_msi" \
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

echo compute test accuracies

python3 -c "from main_test import main; main('scripts/linear/$DATASET/$CONFIG_NAME.yaml')"

echo "done"

########################

EXPERIMENT_NAME="eurosat_a15"
DATASET="eurosat_msi"
BACKBONE="resnet50" 
CONFIG_NAME="frossl_a15"

echo "run $CONFIG_NAME."

python3 -u main_pretrain.py \
    --config-path "scripts/pretrain/eurosat_msi" \
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

echo compute test accuracies

python3 -c "from main_test import main; main('scripts/linear/$DATASET/$CONFIG_NAME.yaml')"

echo "done"


########################

EXPERIMENT_NAME="eurosat_mc"
DATASET="eurosat_msi"
BACKBONE="resnet50" 
CONFIG_NAME="frossl_mc"

echo "run $CONFIG_NAME."

python3 -u main_pretrain.py \
    --config-path "scripts/pretrain/eurosat_msi" \
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

echo compute test accuracies

python3 -c "from main_test import main; main('scripts/linear/$DATASET/$CONFIG_NAME.yaml')"

echo "done"


########################

EXPERIMENT_NAME="eurosat_mv4"
DATASET="eurosat_msi"
BACKBONE="resnet50" 
CONFIG_NAME="frossl_mv4"

echo "run $CONFIG_NAME."

python3 -u main_pretrain.py \
    --config-path "scripts/pretrain/eurosat_msi" \
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

echo compute test accuracies

python3 -c "from main_test import main; main('scripts/linear/$DATASET/$CONFIG_NAME.yaml')"

echo "done"


########################

EXPERIMENT_NAME="eurosat_rt45"
DATASET="eurosat_msi"
BACKBONE="resnet50" 
CONFIG_NAME="frossl_rt45"

echo "run $CONFIG_NAME."

python3 -u main_pretrain.py \
    --config-path "scripts/pretrain/eurosat_msi" \
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

echo compute test accuracies

python3 -c "from main_test import main; main('scripts/linear/$DATASET/$CONFIG_NAME.yaml')"

echo "done"



########   LONGRUNS #########

EXPERIMENT_NAME="eurosat_ivne"
DATASET="eurosat_msi"
BACKBONE="resnet50" 
CONFIG_NAME="ivne"

echo "run $CONFIG_NAME."

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

python3 -c "from main_test import main; main('scripts/linear/$DATASET/$CONFIG_NAME.yaml')"

echo "done"



EXPERIMENT_NAME="eurosat_mmcr"
DATASET="eurosat_msi"
BACKBONE="resnet50" 
CONFIG_NAME="mmcr"

echo "run $CONFIG_NAME."

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

python3 -c "from main_test import main; main('scripts/linear/$DATASET/$CONFIG_NAME.yaml')"

echo "done"



EXPERIMENT_NAME="eurosat_simclr"
DATASET="eurosat_msi"
BACKBONE="resnet50" 
CONFIG_NAME="simclr"

echo "run $CONFIG_NAME."

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

python3 -c "from main_test import main; main('scripts/linear/$DATASET/$CONFIG_NAME.yaml')"

echo "done"