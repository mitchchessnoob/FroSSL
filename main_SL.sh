echo "run frossl SemiSL experiment."

export KAGGLE_USERNAME="michelenarese"
export KAGGLE_KEY="0a593cdaf1707de5e0924de7be17019a"

CONFIGS_PATH=/content/FroSSL/semi_supervised/configs.yaml
AUGMENTS_PATH=/content/FroSSL/scripts/pretrain/office31/augmentations/asymmetric.yaml
KEY="57e49312fc462a736d24abd32cc7891d91258b76"

python3 -u main_semiSL.py \
    --config_path $CONFIGS_PATH \
    --augments_path $AUGMENTS_PATH \
    --key=$KEY

echo "done"
