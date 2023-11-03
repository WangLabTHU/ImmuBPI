#!/bin/bash

# 循环迭代参数
for ((i=0; i<=4; i++))
do
    # 调用Python脚本并传递参数
    python run_main.py -c ./configs/binding_config.yaml -f "$i"
done

# 循环迭代参数
for ((i=0; i<=4; i++))
do
    # 调用Python脚本并传递参数
    python run_main.py -c ./configs/presentation_config.yaml -f "$i"
done

# 循环迭代参数
for ((i=0; i<=9; i++))
do
    # 调用Python脚本并传递参数
    python run_main.py -c ./configs/immunogenicity_config.yaml -f "$i"
done
