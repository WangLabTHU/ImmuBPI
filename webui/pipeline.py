from pathlib import Path
from typing import Any, Callable, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

import data as data_package
import models as model_package
from utils import init_obj


class Ensemble(nn.Module):
    def __init__(self, model_list: List[nn.Module]):
        super().__init__()
        self.model_list = nn.ModuleList(model_list)

    def forward(self, *args, **kwargs):
        outputs = [model(*args, **kwargs) for model in self.model_list]
        outputs = [F.softmax(output, dim=-1) for output in outputs]
        output = sum(outputs) / len(outputs)
        return output


class ImmuBPIPipeline:
    def __init__(
        self,
        model: nn.Module,
        collate_fn: Callable[[List[dict]], dict],
        device: Union[str, int, torch.device] = "cpu",
        need_softmax: bool = True,
    ) -> None:
        self.model = model
        self.collate_fn = collate_fn
        self.device = device
        self.need_softmax = need_softmax

        self.model.eval()
        self.model.to(device)

    def to(self, device: Union[str, int, torch.device]):
        device = torch.device(device)
        self.model.to(device)
        self.device = device
        return self

    @classmethod
    def from_pretrained(
        cls,
        path: Union[str, Path],
        model_ckpt_name: str = "best_model.pth",
        ensemble: bool = False,
    ):
        path = Path(path) if isinstance(path, str) else path
        if not ensemble:
            with open(path / "config.yaml") as f:
                config = yaml.safe_load(f)

            model = init_obj(config["model"], model_package)
            model.load_state_dict(torch.load(path / model_ckpt_name))

            collate_fn = init_obj(config["collate_function"], data_package)
            return cls(model=model, collate_fn=collate_fn)
        else:
            with open(path / "Fold_0" / "config.yaml") as f:
                config = yaml.safe_load(f)

            collate_fn = init_obj(config["collate_function"], data_package)

            model_list = []
            for sub_folder in path.iterdir():
                if not sub_folder.is_dir():
                    continue
                if not sub_folder.stem.startswith("Fold"):
                    continue

                model = init_obj(config["model"], model_package)
                model.load_state_dict(torch.load(sub_folder / model_ckpt_name))
                model_list.append(model)

            model = Ensemble(model_list)
            return cls(model=model, collate_fn=collate_fn, need_softmax=False)

    @torch.no_grad()
    def __call__(
        self,
        epitope: Union[str, List[str]],
        hla: Union[str, List[str]],
        *args: Any,
        **kwds: Any,
    ) -> Any:
        # 0. preprocess
        if isinstance(epitope, str):
            epitope = [epitope]
        if isinstance(hla, str):
            hla = [hla]
        assert len(epitope) == len(hla), "number of of epitope and HLA must in same length"

        # 1. tokenize
        input_dicts = [{"epitope": e, "hla": h} for e, h in zip(epitope, hla)]
        input_dicts = self.collate_fn(input_dicts)

        # 2. inference
        input_dicts = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in input_dicts.items()}
        outputs = self.model(**input_dicts)

        # 3. post process
        if self.need_softmax:
            outputs = F.softmax(outputs, dim=-1)
        outputs = outputs[:, 1]
        outputs = outputs.cpu().numpy()

        return outputs


if __name__ == "__main__":
    pipe = ImmuBPIPipeline.from_pretrained("./RECOMB_models_saved/bigmhc_immu_all/Fold_0")
    print(pipe.model)
    print(pipe.device)
    epitope = ["ALASCMGLIY", "ALEVLQSIPY"]
    HLA = ["HLA-A*01:01"] * 2
    print(pipe(epitope, HLA))
