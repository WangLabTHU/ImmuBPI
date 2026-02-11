import argparse
import logging
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

import data as data_package
import models as model_package
from utils import PerformanceEvaluator, init_obj


# Monkey patch for Attention Record
## In pytorch 1.12+ TransformerEncoder Implements, it call multihead attention module with kwargs - need_weights = False
## to accelerate forward pass. But we need the attention score in this script.
## So we change the multihead attention forward funtion by force the "need_weights" equals True

old_multihead_attention_forward = nn.MultiheadAttention.forward


def new_multihead_attention_forward(*args, **kwargs):
    kwargs["need_weights"] = True
    return old_multihead_attention_forward(*args, **kwargs)


nn.MultiheadAttention.forward = new_multihead_attention_forward


class RecordAttentionWeight(object):
    attn_dict = defaultdict(list)

    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, model: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        if not isinstance(model, nn.MultiheadAttention):
            raise TypeError("Not a MultiheadAttention Layer!")
        self.attn_dict[self.name].append(output[1].cpu().detach())


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
        self.init_save()

    def init_test(self):
        """
        initialize configs for model testing
        load model
        load test dataset
        """

        # init hook
        # self.attention_weight_recorder = RecordAttentionWeight()
        # load model
        self.model = init_obj(self.args.model, model_package)
        self.model.load_state_dict(torch.load(self.args.model_path))
        self.model = self.model.cuda()
        self.model.eval()
        # register hook
        self.register_hook()

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
        hook_list = [
            module
            for module in getattr(self.model, self.args.draw_attn_module).modules()
            if isinstance(module, nn.TransformerEncoderLayer)
        ]
        self.attn_layer_num = len(hook_list)
        self.attn_layer_name_list = []
        for i in range(self.attn_layer_num):
            self.attention_weight_recorder = RecordAttentionWeight(
                name=self.args.draw_attn_module + "layers_" + str(i)
            )
            getattr(self.model, self.args.draw_attn_module).layers[i].self_attn.register_forward_hook(
                self.attention_weight_recorder
            )
            self.attn_layer_name_list.append(self.args.draw_attn_module + "layers_" + str(i))
            print(f"Register hook {self.args.draw_attn_module} layers_ {str(i)}")

    def init_utils(self):
        self.test_evaluator = PerformanceEvaluator()

    def init_save(self):
        """
        create result folder
        """
        self.save_root = Path(self.args.model_path).parent / self.args.test_dataset["args"]["type"]
        print(self.save_root)
        self.save_root.mkdir(parents=True, exist_ok=True)

    def test(self):
        print("*" * 10, "Testing Start", "*" * 10)

        # for saving result and drawing roc curve
        self.res_all = []
        self.label_all = []

        self.res_subset = []
        self.label_subset = []

        for batch, data in enumerate(self.test_dataloader):
            if torch.cuda.is_available():
                data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

            output = self.model(**data)
            prob = torch.softmax(output, dim=-1)[:, 1]

            res = prob.cpu().detach().numpy()
            label = data["label"].cpu().detach().numpy()
            hla_list = data["hla_type"]

            self.test_evaluator.update(label, res)
            self.res_all.extend(res)
            self.label_all.extend(label)

            for idx in range(len(hla_list)):
                if hla_list[idx] == "HLA-A01:01":
                    self.res_subset.append(res[idx])
                    self.label_subset.append(label[idx])

        # {'transformer_encoder_layer_name':[[batch_num],[batch_size],[[L],[L]]]}

        attn_dict = self.attention_weight_recorder.attn_dict
        arrays = []  # noqa: F841

        # concatenate batch
        for k, v in attn_dict.items():
            # arrays.append(torch.cat(v, 0).detach().cpu().numpy()[:,0,:])
            arr = np.concatenate([item.detach().cpu().numpy() for item in v])[:, 0, :]

            # concatenated_array = np.array(arrays)
            np.save(self.save_root / f"{k}_sample_attention_weight.npy", arr)

        self.attn_dict = {
            k: np.mean(torch.cat(v, dim=0).cpu().detach().numpy(), axis=0)[0, :] for k, v in attn_dict.items()
        }
        self.draw_attention_map()

        # save original score
        self.test_dataset.data.insert(loc=1, column="model_score", value=self.res_all)
        save_path = os.path.join(self.save_root, "model_score.csv")
        self.test_dataset.data.to_csv(save_path)
        print("*" * 10, f"Save Result to {save_path}", "*" * 10)

        self.test_evaluator.cal_performance(save_dir=self.save_root)

        print(f"Test Set AUC {self.test_evaluator.auc:.4f} | PRAUC {self.test_evaluator.prauc:.4f}")
        print(f"MCC :{self.test_evaluator.MCC:.4f} | F1_Score:{self.test_evaluator.F1_score:.4f}")
        print(
            f"Test Sensitivity:{self.test_evaluator.sensitivity:.4f} | Specificity:{self.test_evaluator.specificity:.4f}"
        )
        print(f"Test Top20 : {self.test_evaluator.top20_hit} | Top50 : {self.test_evaluator.top50_hit}")

        print("*" * 10, "Testing End", "*" * 10)

    def draw_attention_map(self):
        plt.figure(figsize=(20, 10))

        attn_df = pd.DataFrame.from_dict(self.attn_dict)
        print([np.max(v) for k, v in self.attn_dict.items()])
        plt.title("Attn_Layer Score")

        sns.heatmap(attn_df.T, cmap="Blues", square=True, linewidths=0.25, cbar_kws={"shrink": 0.2})
        # sns.heatmap(attn_df.T,
        # cmap = 'YlGnBu', square = True, linewidths=0.2,cbar_kws={"shrink": .2})

        save_path = self.save_root / "layers_heatmap.png"
        plt.savefig(save_path)
        print("*" * 10, f"Save Result to {save_path}", "*" * 10)

        # print(attn_df.T.mean(axis=0))


def parser_args() -> ConfigArgs:
    parser = argparse.ArgumentParser(description="torchepitope testing")
    parser.add_argument("-p", "--path", type=str, help="models_saved_dir_path, contains yaml and .pkl file")
    parser.add_argument("-m", "--model", default="best_model.pth", type=str, help="model_name")
    parser.add_argument("-t", "--testset", default="TESLA", type=str, help="test_benchmark")
    parser.add_argument("--prs_score", default="", type=str, help="test_benchmark presentation_score")

    parser.add_argument(
        "--draw_attn_module", default="combine_encoder", type=str, help="Module name for plotting attention map"
    )

    args = parser.parse_args()
    config_path = Path(args.path) / "config.yaml"
    assert config_path.is_file(), print("Please input a valid path which contains config.yaml file")
    model_path = Path(args.path) / args.model
    assert config_path.is_file(), print("Model doesn't exist, check again.")

    # load base setting
    with open(config_path, "rt") as f:
        config_dict = yaml.safe_load(f)
    config_args = ConfigArgs(**config_dict)

    # update test setting
    config_args.test_dataset["args"]["type"] = args.testset
    config_args.model_path = model_path
    config_args.prs_score = args.prs_score
    config_args.draw_attn_module = args.draw_attn_module

    return config_args


def main():
    args = parser_args()
    tester = Test(args=args)
    tester.test()


if __name__ == "__main__":
    main()
