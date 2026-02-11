import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import data as data_package
import models as model_package
from utils import PerformanceEvaluator, init_obj


class RecordForwardRepresentation(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.value = []

    def __call__(self, model: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        self.value.append(output.cpu().detach())


class ConfigArgs(object):
    def __init__(self, **arg_dicts) -> None:
        self.__dict__.update(arg_dicts)


class Test:
    def __init__(
        self,
        args: "argparse.Namespace",
    ) -> None:
        self.args = args
        self.init_test()
        logging.debug(f"CUDA: {torch.cuda.is_available()}")
        self.init_utils()

    def init_test(self):
        """
        initialize configs for model testing
        load model
        load test dataset
        """

        # init hook
        # self.attention_weight_recorder = RecordAttentionWeight()
        # load model

        # self.model = init_obj(self.args.model, model_package)
        # self.model.load_state_dict(torch.load(self.args.model_path))
        # self.model = self.model.cuda()
        # self.model.eval()

        # register hook
        # self.register_hook()

        # load dataset
        self.collate_function = init_obj(self.args.collate_function, data_package)
        self.test_dataset = init_obj(self.args.test_dataset, data_package)
        self.test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            collate_fn=self.collate_function,
            drop_last=False,
        )

    def register_hook(self):

        self.forward_transformer_recorder = RecordForwardRepresentation(name="Transformer Encoder Representation")
        getattr(self.model, self.args.draw_attn_module).register_forward_hook(self.forward_transformer_recorder)

        self.forward_projection_recorder = RecordForwardRepresentation(name="Transformer Projection Representation")
        getattr(self.model, "projection")[1].register_forward_hook(self.forward_projection_recorder)

        print("Register hooks Successfully!")

    def init_utils(self):
        self.test_evaluator = PerformanceEvaluator()

    def get_save_path(self, model_path):
        """
        create result folder
        """
        save_root = Path(model_path) / self.args.test_dataset["args"]["type"]
        save_root.mkdir(parents=True, exist_ok=True)
        return save_root

    def test(self):
        print("*" * 10, "Testing Start", "*" * 10)
        self.transformer_value_all_models = []  # type: List[np.ndarray]
        self.projection_value_all_models = []  # type: List[np.ndarray]
        self.projection_layer_all_models = []  # type: List[torch.Linear]

        for single_model_path in self.args.model_path:
            # self.attention_weight_recorder = RecordAttentionWeight()
            self.model = init_obj(self.args.model, model_package)
            self.model.load_state_dict(torch.load(single_model_path))
            self.model = self.model.cuda()
            self.model.eval()
            self.register_hook()
            print("*" * 10, f"Loading Model {single_model_path}", "*" * 10)

            self.res_all = []
            self.label_all = []
            for batch, data in enumerate(self.test_dataloader):
                if torch.cuda.is_available():
                    data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

                output = self.model(**data)  # noqa: F841

            # concatenate batch, [[batch1], [batch2], ... [bacth3]] to [LEN(DATASET), FEATURE_DIM]
            transformer_value = torch.cat(self.forward_transformer_recorder.value, dim=0).cpu().detach().numpy()
            projection_value = torch.cat(self.forward_projection_recorder.value, dim=0).cpu().detach().numpy()

            # only get cls feature
            transformer_value = transformer_value[:, 0, :]
            self.transformer_value_all_models.append(transformer_value)
            self.projection_value_all_models.append(projection_value)
            self.projection_layer_all_models.append(self.model.projection[-1].cpu())

            path = Path(os.path.dirname(single_model_path))
            # 保存单个模型结果
            with open(self.get_save_path(path) / "transformer_feature.npy", "wb") as f:
                np.save(f, transformer_value)
                print("*" * 10, f"Save features to {self.get_save_path(path)}/transformer_feature.npy", "*" * 10)

            with open(self.get_save_path(path) / "projection_feature.npy", "wb") as f:
                np.save(f, projection_value)
                print("*" * 10, f"Save features to {self.get_save_path(path)}/projection_feature.npy", "*" * 10)

        # 综合计算得分
        self.save_root = self.get_save_path(self.args.model_path[0].parent.parent)
        with open(self.save_root / "transformer_feature.npy", "wb") as f:
            self.transformer_value_all_models = np.concatenate(self.transformer_value_all_models, axis=1)
            np.save(f, self.transformer_value_all_models)
            print("*" * 10, f"Save features to {self.save_root}/transformer_feature.npy", "*" * 10)

        with open(self.save_root / "projection_feature.npy", "wb") as f:
            # n_model = len(self.projection_layer_all_models)
            # for i in range(n_model):
            #     features = self.projection_value_all_models[i]
            #     proj_layer = self.projection_layer_all_models[i]
            #     features = self.transform(proj_layer, features)
            #     self.projection_value_all_models[i] = features

            self.projection_value_all_models = np.concatenate(self.projection_value_all_models, axis=1)
            np.save(f, self.projection_value_all_models)

            print("*" * 10, f"Save features to {self.save_root}/projection_feature.npy", "*" * 10)


def parser_args() -> ConfigArgs:
    parser = argparse.ArgumentParser(description="generate model embedding")
    parser.add_argument("-p", "--path", type=str, help="models_saved_dir_path, contains yaml and .pkl file")
    parser.add_argument("-m", "--model", default="best_model.pth", type=str, help="model_name")
    parser.add_argument("-t", "--test_dataset", default="TESLA", type=str, help="test_benchmark")
    parser.add_argument(
        "-e", "--ensemble", default=False, action="store_true", help="aggregate predictions from multiple models"
    )
    parser.add_argument(
        "--draw_attn_module", default="combine_encoder", type=str, help="Module name for plotting attention map"
    )

    args = parser.parse_args()
    model_path_list = []
    if not args.ensemble:
        config_path = Path(args.path) / "config.yaml"
        assert config_path.is_file(), print("Please input a valid path which contains config.yaml file")
        model_path = Path(args.path) / args.model
        assert config_path.is_file(), print("Model doesn't exist, check again.")
        with open(config_path, "rt") as f:
            config_dict = yaml.safe_load(f)
        model_path_list.append(model_path)
    else:
        model_dir_list = os.listdir(Path(args.path))
        for model in model_dir_list:
            config_path = Path(args.path) / Path(model) / "config.yaml"
            if not config_path.is_file():
                continue
            model_path = Path(args.path) / Path(model) / args.model
            if not model_path.is_file():
                continue

            with open(config_path, "rt") as f:
                config_dict = yaml.safe_load(f)
            model_path_list.append(model_path)

    # load base setting

    config_args = ConfigArgs(**config_dict)
    # update test setting
    config_args.test_dataset["args"]["type"] = args.test_dataset
    config_args.model_path = model_path_list
    config_args.draw_attn_module = args.draw_attn_module

    return config_args


def main():
    args = parser_args()
    tester = Test(args=args)
    tester.test()


if __name__ == "__main__":
    main()
