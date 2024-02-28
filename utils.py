import os
import re
import json
from typing import List, Union
from datasets import Dataset


def read_file(fp: str) -> List[str]:
    assert os.path.exists(fp)
    with open(fp, 'r', encoding="utf-8") as f:
        return f.readlines()


def create_dataset(code_fp: str,
                   summ_fp: str,
                   code_score_fp: Union[str, None] = None,
                   summ_score_fp: Union[str, None] = None
                   ) -> Dataset:
    input_dict = {}
    for fp in [code_fp, summ_fp]:
        assert os.path.exists(fp)
    input_dict['code'] = read_file(code_fp)
    input_dict['summ'] = read_file(summ_fp)
    if code_score_fp and summ_score_fp:
        for fp in [code_score_fp, summ_score_fp]:
            assert os.path.exists(fp)
        input_dict['code_score'] = \
            [float(x) for x in read_file(code_score_fp)]
        input_dict['summ_score'] = \
            [float(x) for x in read_file(summ_score_fp)]
    return Dataset.from_dict(input_dict)


def check_folder(dir_path: str):
    """检查文件夹dir_path是否为空"""
    assert os.path.exists(dir_path)
    lst = os.listdir(dir_path)
    if lst:
        return True
    return False


def create_test_dataset(
    src_fp: str,
    tgt_fp: str
) -> Dataset:
    for fp in [src_fp, tgt_fp]:
        assert os.path.exists(fp)
    input_dict = {
        "src_list": read_file(src_fp),
        "tgt_list": read_file(tgt_fp)
    }
    return Dataset.from_dict(input_dict)


def recover(s: str):
    ans = ""
    s = s.strip('\n')
    n = len(s)
    for i, c in enumerate(s):
        if c.isalpha():
            ans += c
        elif c == " ":
            ans += c
        else:
            if i > 0:
                ans += " "
            ans += c
            if i < n - 1:
                ans += " "
    pattern = re.compile(r"[ ]+")
    ans = re.sub(pattern, " ", ans)
    return ans


def dump_json(
    items,
    fp: str
):
    with open(fp, 'w') as f:
        json.dump(obj=items, fp=f, indent=4)
        
        
def load_json(
    fp: str
):
    with open(fp, 'r') as f:
        return json.load(f)

