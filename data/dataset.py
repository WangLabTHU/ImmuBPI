import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from typing_extensions import TypedDict

from data.data_utils import adaptive_transfer_name


# from data.constant import MAX_EPITOPE_LEN, MAX_HLA_LEN
class SampleDict(TypedDict):
    peptide: str
    fold: int
    length: int
    HLA: str
    immunogenicity: int
    hla_seq: str


class ImmuBPIDataset(Dataset):
    def __init__(
        self,
        type: str,
        task_type: str,  # ["binding", "presentation", "immunogenicity"]
        prefix: str,
        fold: Optional[int] = None,
        train_dataset_dir: Optional[str] = None,
        dataset_name: Optional[List[str]] = None,
        hla_inner_balanced: bool = False,
        given_hla_peptides: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> None:
        self.type = type
        self.task_type = task_type
        self.prefix = Path(prefix)
        self.fold = fold
        self.train_dataset_dir = train_dataset_dir
        self.dataset_name = dataset_name

        self.hla_inner_balanced = hla_inner_balanced
        self.hla_ratio_cache = {}  # type: Dict[str, List[float, float]]

        self.given_hla_peptides = given_hla_peptides
        if self.given_hla_peptides:
            self.make_data_from_command_line()
        else:
            self.load_data_from_file()
            self.load_data_from_csv()

    def load_data_from_file(self):

        if self.type in ["train", "val"]:
            if self.task_type in ["binding", "presentation"]:
                data_dir = os.path.join(
                    self.prefix, self.train_dataset_dir, self.type, self.type + "_data_fold" + str(self.fold) + ".csv"
                )
                self.data = pd.read_csv(data_dir)
            elif self.task_type == "immunogenicity":
                # For immunogenicity task dataset
                data_dir = os.path.join(self.prefix, self.train_dataset_dir)
                self.data = pd.read_csv(data_dir)
                if self.type == "train":
                    self.data = self.data[self.data["fold"] != self.fold]
                else:
                    self.data = self.data[self.data["fold"] == self.fold]
            else:
                raise ValueError("Invalid Task type", self.task_type)
            print(f"Load Training Dataset {data_dir}")

        else:
            df = pd.read_csv(self.prefix / "dataset_register_table.csv")
            self.dataset_dict = dict(zip(df.dataset_name, df.file_path))
            dir = self.dataset_dict[self.type]
            self.data = pd.read_csv(self.prefix / dir)
            print(f"Load Testing Dataset {self.prefix / dir}")

        self.hla_type2seq = np.load(self.prefix / "const/hla_type2seq.npy", allow_pickle=True).item()

    def find_proxy(self, HLA):
        # for unknown seq HLA, find similar one
        # e.g. HLA-B07:01 HLA-B07:02
        for k, v in self.hla_type2seq.items():
            if HLA[0:7] == k[0:7]:
                print(f"replace {HLA} with {k}")
                return k

        return ""

    def make_data_from_command_line(self):
        self.hla_type2seq = np.load(self.prefix / "const/hla_type2seq.npy", allow_pickle=True).item()

        self._meta_data = [{"peptide": item[0], "HLA": item[1]} for item in self.given_hla_peptides]
        print(f"Num of data: {len(self._meta_data)}")
        print(self._meta_data)

        self.peptide_list = []
        self.hla_type_list = []
        self.hla_seq_list = []

        for index, sample_dict in enumerate(self._meta_data):
            try:
                standard_name = adaptive_transfer_name(sample_dict["HLA"])
                self._meta_data[index]["hla_seq"] = self.hla_type2seq[standard_name]
                # self.hla_seq_list.append(self.hla_type2seq[standard_name])
                # self.hla_type_list.append(standard_name)
                # self.peptide_list.append(sample_dict["peptide"])
            except Exception:
                print(f"Unfound HLA type {sample_dict['HLA']}")

        self.index = list(range(len(self._meta_data)))

    def load_data_from_csv(self):

        self._meta_data = list(self.data.T.to_dict().values())  # type: List[SampleDict]
        print(f"Original Dataset Num of Data: {len(self._meta_data)}")
        filtered_indices = []
        for index, sample_dict in enumerate(self._meta_data):
            try:
                standard_name = adaptive_transfer_name(sample_dict["HLA"])
                sample_dict["hla_seq"] = self.hla_type2seq[standard_name]
                sample_dict["HLA"] = standard_name
                filtered_indices.append(index)
            except Exception:
                standard_name = self.find_proxy(adaptive_transfer_name(sample_dict["HLA"]))
                if standard_name:
                    sample_dict["hla_seq"] = self.hla_type2seq[standard_name]
                    sample_dict["HLA"] = sample_dict["HLA"]
                    filtered_indices.append(index)
                else:
                    print(f"{sample_dict['HLA']} unfound")

        self._meta_data = [self._meta_data[index] for index in filtered_indices]
        self.index = list(range(len(self._meta_data)))

        print(f"After filtering invalid hla, remain {len(self._meta_data)}")

    def cal_hla_balance_ratio(self, hla: str) -> List[int]:
        #  Note: this function is used and specific for EL training, since for
        #  some alleles, they have highly unbalanced pos/neg ratio.
        if hla in self.hla_ratio_cache:
            return self.hla_ratio_cache[hla]

        pos = 0
        neg = 0
        for sample_dict in self._meta_data:
            hla_iter = sample_dict["HLA"]
            label = sample_dict["label"]
            if hla_iter != hla:
                continue
            if label == 1:
                pos += 1
            else:
                neg += 1
        pos = neg / pos  # neg as unit 1
        pos = 10.0 if pos > 10 else (pos if pos >= 0.1 else 0.1)

        self.hla_ratio_cache[hla] = [1.0, pos]
        return self.hla_ratio_cache[hla]

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_dict = self._meta_data[index]
        peptide = sample_dict["peptide"]
        hla = sample_dict["hla_seq"]
        hla_type = sample_dict["HLA"]

        if self.task_type in ["binding", "presentation"]:
            return_dict = {
                "epitope": peptide,
                "hla": hla,
                "hla_type": adaptive_transfer_name(hla_type),
                "label": sample_dict.get("label", sample_dict.get("immunogenicity", -1)),
            }
        else:
            return_dict = {
                "epitope": peptide,
                "hla": hla,
                "hla_type": adaptive_transfer_name(hla_type),
                "label": sample_dict.get("immunogenicity", -1),
            }

        if self.hla_inner_balanced:
            weight = self.cal_hla_balance_ratio(hla_type)
            return_dict["weight"] = weight[sample_dict.get("label", -1)]

        return return_dict

    def __len__(self):
        return len(self.index)


if __name__ == "__main__":
    import os

    train_params = {
        "type": "train",
        "prefix": "./dataset/",
        "fold": 0,
        "train_dataset_dir": "./NetMHCpanEL/",
    }

    train_set = ImmuBPIDataset(**train_params)
    for idx, data in enumerate(train_set):
        if idx > 10:
            break
        print(data)
