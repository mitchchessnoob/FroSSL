echo "run frossl SemiSL experiment."

export KAGGLE_USERNAME="michelenarese"
export KAGGLE_KEY=#CENSORED

CONFIGS_PATH=semi_supervised/configs.yaml


python3 -u main_SL_morestandard.py \
    --config_path $CONFIGS_PATH \

echo "done"
