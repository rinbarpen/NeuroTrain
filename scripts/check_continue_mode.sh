python main.py -c 'configs/single/train.template.toml' --task_name 'continue' --run_id 0 --train --test --predict -e 200
# wait 30
# Ctrl-C
python main.py -c 'output/continue/0/train/config.toml' --continue_id '0' # select 0
# python main.py -c 'output/continue/0/train/config.toml' --continue_id 'auto' # automatically select run_id
