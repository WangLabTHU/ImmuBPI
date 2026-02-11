"""
retrive model results
given peptide & HLA pairs
"""

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader

import data as data_package
import models as model_package
from utils import init_obj


class ConfigArgs(object):
    def __init__(self, **arg_dicts) -> None:
        self.__dict__.update(arg_dicts)


class Retrive_Results:
    def __init__(
        self,
        args: "argparse.Namespace",
    ) -> None:
        self.args = args
        self.init_test()
        logging.debug(f"CUDA: {torch.cuda.is_available()}")

    def init_test(self):
        """
        initialize configs for model testing
        load model
        load test dataset
        """

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

    def test(self):
        self.res_all_all_models = []
        for single_model_path in self.args.model_path:
            self.model = init_obj(self.args.model, model_package)
            self.model.load_state_dict(torch.load(single_model_path))
            self.model = self.model.cuda()
            self.model.eval()

            self.res_all = []
            self.peptide_all = []
            self.label_all = []
            self.hla_type_all = []

            for batch, data in enumerate(self.test_dataloader):
                if torch.cuda.is_available():
                    data = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

                output = self.model(**data)
                prob = torch.softmax(output, dim=-1)[:, 1]
                res = prob.cpu().detach().numpy()
                # res = output.cpu().detach().numpy()[:, 0]

                self.res_all.extend(res)
                self.hla_type_all.extend(data["hla_type"])
                self.peptide_all.extend(data["epitope"].cpu().detach().numpy())
                # self.label_all.extend(label)

            self.res_all_all_models.append(self.res_all)

        if len(self.args.model_path) >= 1:
            print("*" * 10, "Calculating Model Ensemble Performance", "*" * 10)
            spear_cor = 0
            for item1 in self.res_all_all_models:
                for item2 in self.res_all_all_models:
                    if item1 != item2:
                        spear_cor += stats.spearmanr(item1, item2).statistic

            if len(self.args.model_path) > 1:
                print(
                    "Mean spearman cor",
                    spear_cor / (len(self.res_all_all_models) * (len(self.res_all_all_models) - 1)),
                )

            self.ensemble_score = np.mean(np.array(self.res_all_all_models), axis=0)
            self.ensemble_score_std = np.std(np.array(self.res_all_all_models), axis=0)
            print("Final ensemble score(Avg)")
            print(self.ensemble_score)
            # print('Final ensemble score(Std)')
            # print(self.ensemble_score_std)

            df_saved = pd.DataFrame(  # noqa: F841
                data={"HLA": self.hla_type_all, "peptide": self.peptide_all, "model_score": self.ensemble_score}
            )
            # df_saved.to_csv('retrieve_model_results.csv')


def parser_args() -> ConfigArgs:
    parser = argparse.ArgumentParser(description="torchepitope testing")
    parser.add_argument(
        "-p",
        "--path",
        default="./models_saved/DeepNeo_MaskHLA_MaskAnchor_10FOLD",
        type=str,
        help="models_saved_dir_path, contains yaml and .pkl file",
    )
    parser.add_argument("-m", "--model", default="best_model.pth", type=str, help="model_name")
    parser.add_argument(
        "-e", "--ensemble", default=False, type=bool, help="aggregate predictions from multiple models"
    )

    parser.add_argument("--peptide", help="peptide", nargs="+")
    parser.add_argument("--HLA", default=["HLA-A24:02"], nargs="+", help="hla")

    args = parser.parse_args()
    if not args.ensemble:
        config_path = Path(args.path) / "config.yaml"
        assert config_path.is_file(), print("Please input a valid path which contains config.yaml file")
        model_path = Path(args.path) / args.model
        assert config_path.is_file(), print("Model doesn't exist, check again.")
        with open(config_path, "rt") as f:
            config_dict = yaml.safe_load(f)
        model_path_list = [model_path]
    else:
        model_dir_list = os.listdir(Path(args.path))
        model_path_list = []
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
        print(f"total {len(model_path_list)} Models")

    # load base setting

    config_args = ConfigArgs(**config_dict)
    # update test setting
    config_args.model_path = model_path_list

    # make prediction pairs
    # Example 1 (with broadcast for HLA):
    # peptide: AAAAA, BBBBB
    # HLA: HLA-A02:01
    # pairs: [[AAAAA, HLA-A02:01], [BBBBB, HLA-A02:01]]
    # Example 2:
    # peptide: AAAAA, BBBBB
    # HLA: HLA-A02:01, HLA-A24:02
    # pairs: [[AAAAA, HLA-A02:01], [BBBBB, HLA-A24:02]]

    if len(args.HLA) == 1:
        config_args.test_dataset["args"]["given_hla_peptides"] = [[item, args.HLA[0]] for item in args.peptide]
    else:
        assert len(args.peptide) == len(args.HLA)
        config_args.test_dataset["args"]["given_hla_peptides"] = [
            [args.peptide[idx], args.HLA[idx]] for idx in range(len(args.peptide))
        ]

    return config_args


def main():
    args = parser_args()
    tester = Retrive_Results(args=args)
    tester.test()


if __name__ == "__main__":
    main()
