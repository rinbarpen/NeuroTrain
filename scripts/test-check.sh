#!/bin/bash
python main.py -c configs/single/train-template.toml --check --task test-auto 

python main.py -c configs/single/train-drive.toml --train --test --predict --task test-auto 
