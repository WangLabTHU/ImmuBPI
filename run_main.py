import argparse

import yaml  # type: ignore
import torch
import trainer as trainer_package
import models as model_package
import data as data_package
import loss as loss_package
from utils import init_obj, setup_seed

class ConfigArgs(object):
    def __init__(self, **arg_dicts) -> None:
        self.__dict__.update(arg_dicts)
        # TODO recursive transfer 

def parser_args() -> ConfigArgs:
    parser = argparse.ArgumentParser("torchepitope training and testing")
    parser.add_argument("-c", "--config", type=str, default='./configs/config_template.yaml', help="path to .yaml config file")
    parser.add_argument("-f", "--fold", type=int, default=0, help="dataset fold")

    args = parser.parse_args()
    
    with open(args.config, "rt") as f:
        config_dict = yaml.safe_load(f)

    config_args = ConfigArgs(**config_dict)

    config_args.train_dataset['args']['fold'] = args.fold
    config_args.valid_dataset['args']['fold'] = args.fold

    return config_args

def main():
    args = parser_args()

    setup_seed(getattr(args, 'seed', 2)) 
    print(f"Set up seed {getattr(args, 'seed', 2)}")

    # TODO refactor test as scripts/notebooks
    model = init_obj(args.model, model_package)
    collate_function = init_obj(args.collate_function, data_package)
    train_dataset = init_obj(args.train_dataset, data_package)
    valid_dataset = init_obj(args.valid_dataset, data_package)
    criterion = init_obj(args.loss, loss_package)
    
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    trainer = trainer_package.Trainer(
        args, 
        model=model,
        collate_function=collate_function,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        criterion=criterion
    )

    trainer.train()

if __name__ == "__main__":
    main()
