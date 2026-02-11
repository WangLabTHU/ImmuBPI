import numpy as np


AA_dict = np.load("./dataset/const/AA_dict.npy", allow_pickle=True).item()  # type: dict


def seq2vocab(seq):
    vocab_list = [AA_dict.get(aa, 20) for aa in seq]
    return vocab_list


def counts(str1, str2):
    if str1 == str2:
        return -1
    same = sum([c1 == c2 for c1, c2 in zip(str1, str2)])
    return same


def counts_nonanchor(str1, str2):
    if str1 == str2:
        return -1
    same = sum([c1 == c2 for c1, c2 in zip(str1[3:-2], str2[3:-2])])
    return same


def adaptive_transfer_name(hla_type):
    """
    Target Format: HLA-A01:01
    """
    # substrings_list = ['HLA', 'A*', 'B*', 'C*']
    # if not any(substring in hla_type for substring in substrings_list):
    #     return hla_type

    if len(hla_type) == 10 and "*" in hla_type:
        # HLA-A*0101  HLA-A01:01
        new_hla_type = hla_type[0:5] + hla_type[6:8] + ":" + hla_type[8:]
    elif len(hla_type) == 10 and hla_type[5] == "-":
        # HLA-A-0201  HLA-A01:01
        new_hla_type = hla_type[0:5] + hla_type[6:8] + ":" + hla_type[8:]
    elif len(hla_type) == 5:
        # A0101  HLA-A01:01
        new_hla_type = "HLA-" + hla_type[0:3] + ":" + hla_type[3:]
    elif len(hla_type) == 6:
        new_hla_type = "HLA-" + hla_type
    elif len(hla_type) == 7:
        # A*01:01 HLA-A01:01
        new_hla_type = "HLA-" + hla_type[0:1] + hla_type[2:]
    elif len(hla_type) == 11:
        # HLA-A*01:01 HLA-A01:01
        new_hla_type = hla_type[0:5] + hla_type[6:]
    else:
        new_hla_type = hla_type

    return new_hla_type


# TODO refactor with regex to improve the readbility
"""
Different HLA type string formats and length
which can be converted to standard format by counting length and switch to different convert logic

A0101 5
HLA-A01:01 10
HLA-A*2402 10
HLA-A*01:01  11
A02:01   6
standard HLA-A01:41
"""


def adaptive_transfer(source, target):
    """adaptive transfer for different hla type string"""
    assert isinstance(target, int)

    if target == 5:
        if len(source) == 10 and "*" not in source:
            res = source[4:7] + source[8:]
            return res
        elif len(source) == 10 and "*" in source:
            res = source[4] + source[6:]
            return res
        elif len(source) == 11:
            res = source[4] + source[6:8] + source[9:]
            return res
        elif len(source) == 6:
            res = "HLA-" + source
            return res
        else:
            return source
