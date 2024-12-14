echo "run frossl SemiSL experiment."

export KAGGLE_USERNAME="michelenarese"
export KAGGLE_KEY=#CENSORED

CONFIGS_PATH=semi_supervised/configs.yaml
AUGMENTS_PATH=scripts/pretrain/office31/augmentations/asymmetric.yaml


python3 -u main_SL_morestandard.py \
    --config_path $CONFIGS_PATH \
    --augments_path $AUGMENTS_PATH \

echo "done"
