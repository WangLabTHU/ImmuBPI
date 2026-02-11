from typing import Any, Dict, List

import torch

from data.data_utils import seq2vocab


class SeqDictCollateFn(object):
    def __init__(
        self,
        seq_keys: List[str],
        seq_max_lens: List[int],
        mask_HLA: bool = False,
        mask_epitope: bool = False,
        mask_anchor: bool = False,
    ) -> None:
        self.seq_keys = seq_keys
        self.seq_max_lens = seq_max_lens
        self.mask_HLA = mask_HLA
        self.mask_epitope = mask_epitope
        self.mask_anchor = mask_anchor

        if self.mask_HLA and self.mask_epitope:
            print("Warning, both HLA and epitope are masked!")

    def __call__(self, batch: List[Dict[str, Any]]) -> dict:
        batch_dict = {key: [item[key] for item in batch] for key in batch[0]}  # to Dict[list]
        return_dict = {}

        for k, v in batch_dict.items():
            if k in self.seq_keys:
                max_len = self.seq_max_lens[self.seq_keys.index(k)]
                if isinstance(v[0], str):  # normal sequence as a List of str
                    if k == "epitope":
                        if self.mask_epitope:
                            padding_mask = torch.ones((len(v), max_len), dtype=torch.bool)
                            return_dict[f"{k}_padding_mask"] = padding_mask
                        else:
                            padding_mask = torch.zeros((len(v), max_len), dtype=torch.bool)
                            for idx, seq in enumerate(v):
                                padding_mask[idx, len(seq) :] = True
                                if self.mask_anchor:
                                    padding_mask[idx, 1] = True
                                    padding_mask[idx, len(seq) - 1] = True

                                # if self.mask_anchor and random.randint(0, 1):
                                #     padding_mask[idx, 1] = True
                                # if self.mask_anchor and random.randint(0, 1):
                                #     padding_mask[idx, len(seq)-1] = True

                            return_dict[f"{k}_padding_mask"] = padding_mask

                    if k == "hla":
                        if self.mask_HLA:
                            padding_mask = torch.ones((len(v), max_len), dtype=torch.bool)
                            return_dict[f"{k}_padding_mask"] = padding_mask
                        else:
                            padding_mask = torch.zeros((len(v), max_len), dtype=torch.bool)
                            for idx, seq in enumerate(v):
                                padding_mask[idx, len(seq) :] = True
                            return_dict[f"{k}_padding_mask"] = padding_mask

                    seq_ids = [seq2vocab(seq.ljust(max_len, "*")) for seq in v]

                elif isinstance(v[0][0], str):
                    # TODO 支持任意层的List[List[...[str]]], by 查找深度+partial + map

                    # padding_mask (torch.Tensor, dtype=torch.bool): in shape [B, NUM_WT, L]
                    padding_mask = torch.zeros((len(v), len(v[0]), max_len), dtype=torch.bool)
                    for idx, seq in enumerate(v):
                        for wt_idx in range(len(v[0])):
                            padding_mask[idx, wt_idx, len(v[idx][wt_idx]) :] = True
                    return_dict[f"{k}_padding_mask"] = padding_mask

                    seq_ids = [[seq2vocab(seq.ljust(max_len, "*")) for seq in seq_group] for seq_group in v]

                else:
                    raise Exception("unspported sequence collate")
                return_dict[k] = torch.tensor(seq_ids, dtype=torch.long)
            else:
                try:
                    v = torch.tensor(v)
                    return_dict[k] = v
                except Exception:
                    return_dict[k] = v

        return return_dict
