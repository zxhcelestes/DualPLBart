from typing import List, Union

from torch.utils.data import Dataset


class DualDataset(Dataset):
    def __init__(self,
                 code_list: List[str],
                 summ_list: List[str],
                 code_score_list: Union[List[float], None] = None,
                 summ_score_list: Union[List[float], None] = None) -> None:
        super(DualDataset, self).__init__()
        assert len(code_list) == len(summ_list)
        self.use_dual = False
        if code_score_list and summ_score_list:
            assert len(code_score_list) == len(summ_score_list) == len(
                code_list)
            self.use_dual = True
        self.code_list = code_list
        self.summ_list = summ_list
        self.code_score_list = code_score_list
        self.summ_score_list = summ_score_list

    def __getitem__(self, index):
        # return index for additional information
        if self.use_dual:
            return self.code_list[index], \
                   self.summ_list[index], \
                   self.code_score_list[index], \
                   self.summ_score_list[index]
        else:
            return self.code_list[index], self.summ_list[index]

    def __len__(self):
        return len(self.code_list)
