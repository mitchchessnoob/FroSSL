echo "run frossl SemiSL experiment."

export KAGGLE_USERNAME="michelenarese"
export KAGGLE_KEY=#CENSORED

CONFIGS_PATH=semi_supervised/configs.yaml
AUGMENTS_PATH=semi_supervised/asymmetric.yaml


python3 -u main_semiSL.py \
    --config_path $CONFIGS_PATH \
    --augments_path $AUGMENTS_PATH \


echo "done"
