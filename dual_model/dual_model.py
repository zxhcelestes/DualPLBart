from typing import Union, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PLBartTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import Seq2SeqLMOutput

from dual_model.PLBartForConditionalGeneration import PLBartForConditionalGeneration
# from PLBartForConditionalGeneration import PLBartForConditionalGeneration


class DualModel(PreTrainedModel):
    def __init__(self,
                 config: PretrainedConfig = None,
                 cs_model: Union[None, PLBartForConditionalGeneration] = None,
                 cg_model: Union[None, PLBartForConditionalGeneration] = None,
                 cs_tokenizer: Union[None, PLBartTokenizer] = None,
                 cg_tokenizer: Union[None, PLBartTokenizer] = None,
                 device: Union[torch.device, None] = None,
                 use_dual: bool = False,
                 use_attn: bool = False,
                 lambda_attn: float = 0.01,
                 lambda_dual: float = 0.01):
        config = self.init_config(config)
        super().__init__(config)
        if not isinstance(device, torch.device):
            self._device = torch.device("cuda:0") if torch.cuda.is_available() \
                else torch.device("cpu")
        else:
            self._device = device
        self.cs_model = self.init_model(cs_model, "CS").to(self._device)
        self.cg_model = self.init_model(cg_model, "CG").to(self._device)
        self.cs_tokenizer = self.init_tokenizer(cs_tokenizer, "CS")
        self.cg_tokenizer = self.init_tokenizer(cg_tokenizer, "CG")
        self.use_dual = use_dual
        self.use_attn = use_attn
        self.lambda_dual = lambda_dual
        self.lambda_attn = lambda_attn
        
    def init_config(self,
                    config: Union[None, PretrainedConfig] = None):
        if config is None:
            print("Config is not provided, init base one with `Pretrained_CS_Model`")
            return self.init_model().config
        else:
            return config

    def init_model(self,
                   model: Union[None, PLBartForConditionalGeneration] = None,
                   text: str = ""):
        if model is None:
            print(f"{text} Model is not provided, init base one.")
            return PLBartForConditionalGeneration.from_pretrained(
                "./pretrained_cache/plbart-base")
        return model

    def init_tokenizer(self,
                       tokenizer: Union[None, PLBartTokenizer] = None,
                       text: str = ""):
        if tokenizer is None:
            print(f"{text} tokenizer is not Provided, init base one.")
            return PLBartTokenizer.from_pretrained(
                "./pretrained_cache/plbart-base")
        return tokenizer

    @staticmethod
    def run_base_model(input_text: Union[str, List[str]],
                       target_text: Union[str, List[str]],
                       tokenizer: PLBartTokenizer,
                       model: PLBartForConditionalGeneration,
                       padding_idx: int):
        if isinstance(input_text, str):
            input_text = [input_text]
        if isinstance(target_text, str):
            target_text = [target_text]
        to_inp = tokenizer(input_text,
                           text_target=target_text,
                           return_tensors='pt',
                           padding=True,
                           max_length=256,
                           truncation=True)
        to_inp['output_attentions'] = True
        to_inp['return_dict'] = True
        to_inp['padding_idx'] = padding_idx
        device = model.device
        for k, v in to_inp.items():
            if isinstance(v, torch.Tensor):
                to_inp[k] = v.to(device)
        return to_inp, model(**to_inp)

    @staticmethod
    def log_cond_prob(logits: torch.Tensor):
        """
        用于求每个句子的条件概率，经过log处理。
        Args:
            logits (torch.Tensor[batch_size, max(tgt_len)+1, vocab_size]): 每个输出token的概率分布。
            讨论每一个样例logit[len, vocab_size]，理论上要求 
            $P = \prod_{i=0}^{k-1} P_i, P_i=\frac{max(logit[i])}{sum(logit[i])}$
            由于概率值不大，累乘很可能爆零，使用用$\log$改累乘为累加，返回累加值，即：
            $\log \prod_{i=0}^{k-1} P_i = \Sigma_{i=0}^{k-1} \log P_i$
            可以通过softmax简化$P_i$，即：
            $soft_logits = torch.softmax(logits), p_i = soft_logit[i]
        """
        bsz = logits.size()[0]
        ans = torch.tensor([0. for _ in range(bsz)])
        soft_logits = torch.softmax(logits, -1)
        for i, sen in enumerate(soft_logits):
            log_sum = 0
            for token in sen:
                log_sum += token.max().log()
            ans[i] = log_sum
        return ans

    @staticmethod
    def kl_div(a: torch.Tensor, b: torch.Tensor, mask: torch.BoolTensor,
               length: List[int]):
        epsilon = 1e-8
        a = a.masked_fill(mask, -(1e+8))
        b = b.masked_fill_(mask, -(1e+8))
        a = F.softmax(a, 2) + epsilon
        b = F.softmax(b, 2) + epsilon
        x_a = a * torch.log(a / ((b + a) / 2))
        x_b = b * torch.log(b / ((b + a) / 2))
        x_a = x_a.masked_fill_(mask, 0)
        x_b = x_b.masked_fill_(mask, 0)
        x_a = torch.sum(x_a, 2)
        x_b = torch.sum(x_b, 2)
        x_a = torch.sum(x_a, 1) / torch.FloatTensor(length).cuda()
        x_b = torch.sum(x_b, 1) / torch.FloatTensor(length).cuda()
        kl_div = x_a + x_b
        kl_div = kl_div / 2
        return kl_div

    @staticmethod
    def cross_mask(mask: torch.Tensor, device: torch.dtype, tgt_len: int):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        bsz, src_len = mask.size()
        tgt_len = tgt_len if tgt_len is not None else src_len

        expanded_mask = mask[:, None, None, :].expand(
            bsz, 1, tgt_len, src_len).bool().squeeze().to(device)

        return ~expanded_mask

    def forward(self,
                code: List[str],
                summ: List[str],
                code_score: Union[None, List[float], torch.Tensor] = None,
                summ_score: Union[None, List[float], torch.Tensor] = None,
                ) -> Seq2SeqLMOutput:
        cs_inp, cs_out = self.run_base_model(code, summ, self.cs_tokenizer,
                                             self.cs_model, self.cs_tokenizer.pad_token_id)
        cg_inp, cg_out = self.run_base_model(summ, code, self.cg_tokenizer,
                                             self.cg_model, self.cs_tokenizer.pad_token_id)

        out = Seq2SeqLMOutput()
        out['loss'] = cs_out.loss + cg_out.loss
        
        if self.use_dual:
            assert all(item is not None for item in [code_score, summ_score])
            cs_prob = self.log_cond_prob(cs_out.logits).to(
                self._device)  # log [ P(summ | code) ]
            cg_prob = self.log_cond_prob(cg_out.logits).to(
                self._device)  # log [ p(code | summ) ]
            code_score = torch.tensor(code_score).float() \
                if isinstance(code_score, list) else code_score
            summ_score = torch.tensor(summ_score).float() \
                if isinstance(summ_score, list) else summ_score
            code_score.requires_grad = False
            code_score = code_score.to(self._device)
            summ_score.requires_grad = False
            summ_score = summ_score.to(self._device)
            loss_prob = (code_score + cs_prob - summ_score -
                         cg_prob).square().sum()
            # 条件概率损失
            out['loss'] += self.lambda_dual * loss_prob
            out['loss_prob'] = loss_prob

        if self.use_attn:
            # List[len=num_layers, Tensor[bsz, tgt_len, src_len]]
            cs_attn = [mat.sum(1) for mat in cs_out.cross_attentions]
            cg_attn = [mat.sum(1)
                       for mat in cg_out.cross_attentions][::-1]  # 逆序对应
            code_len = [len(c) for c in code]
            summ_len = [len(s) for s in summ]
            cs_attn_mask = self.cross_mask(cs_inp.attention_mask, self._device,
                                           cs_inp.labels.shape[-1])
            cs_attn_loss_lst = [
                self.kl_div(mat_cs, mat_cg.transpose(1, 2), cs_attn_mask,
                            code_len)
                for mat_cs, mat_cg in zip(cs_attn, cg_attn)
            ]
            cs_attn_loss = (sum(cs_attn_loss_lst) / len(cs_attn_loss_lst)).mean()
            cg_attn_mask = self.cross_mask(cg_inp.attention_mask, self._device,
                                           cg_inp.labels.shape[-1])
            cg_attn_loss_lst = [
                self.kl_div(mat_cg, mat_cs.transpose(1, 2), cg_attn_mask,
                            summ_len)
                for mat_cs, mat_cg in zip(cs_attn, cg_attn)
            ]
            cg_attn_loss = (sum(cg_attn_loss_lst) / len(cg_attn_loss_lst)).mean()
            attn_loss = cs_attn_loss + cg_attn_loss
            # 注意力损失
            out['loss'] += self.lambda_attn * attn_loss
            out['attn_loss'] = attn_loss
            
        return out

    @ staticmethod
    def base_generate(model: PLBartForConditionalGeneration,
                      tokenizer: PLBartTokenizer,
                      inp: Union[List[str], str],
                      num_beams: int = 1
                      ) -> List[str]:
        if isinstance(inp, str):
            inp = [inp]
        device = next(model.parameters()).device
        tgt_lang = tokenizer.tgt_lang
        to_inp = tokenizer(inp,
                           return_tensors='pt',
                           padding=True,
                           max_length=256,
                           truncation=True)
        for k, v in to_inp.items():
            if isinstance(v, torch.Tensor):
                to_inp[k] = v.to(device)
        out = model.generate(
            **to_inp,
            max_length=256,
            decoder_start_token_id=tokenizer.lang_code_to_id[tgt_lang],
            num_beams=num_beams
        )
        return tokenizer.batch_decode(out, skip_special_tokens=True)

    def cs_generate(self,
                    code: Union[List[str], str],
                    num_beams: int = 1) -> List[str]:
        return self.base_generate(
            model=self.cs_model,
            tokenizer=self.cs_tokenizer,
            inp=code,
            num_beams=num_beams
        )

    def cg_generate(self,
                    summ: Union[List[str], str],
                    num_beams: int = 1) -> List[str]:
        return self.base_generate(
            model=self.cg_model,
            tokenizer=self.cg_tokenizer,
            inp=summ,
            num_beams=num_beams
        )


if __name__ == "__main__":
    cs_model = PLBartForConditionalGeneration.from_pretrained(
            "../pretrained_cache/plbart-base")
    cg_model = PLBartForConditionalGeneration.from_pretrained(
            "../pretrained_cache/plbart-base")
    cs_tokenizer = PLBartTokenizer.from_pretrained(
        "../pretrained_cache/plbart-base",
        src_lang='__java__',
        tgt_lang="__en_XX__")
    cg_tokenizer = PLBartTokenizer.from_pretrained(
        "../pretrained_cache/plbart-base",
        src_lang='__en_XX__',
        tgt_lang="__java__")
    mdl = DualModel(config=cs_model.config,
                    cs_model=cs_model,
                    cg_model=cg_model,
                    cs_tokenizer=cs_tokenizer,
                    cg_tokenizer=cg_tokenizer,
                    use_dual=True,
                    device=torch.device("cuda:0"))
    loss = mdl(["a b c", "d e f g"], ['j q k s p', 'x y z lo ve you a a x x'], [0.002, 0.01], [0.520, 0.525]).loss
    print(loss)
    loss.backward()
