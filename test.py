import argparse
import time
from typing import Union
import os

import torch
from datasets import Dataset
from transformers import PLBartTokenizer, PretrainedConfig

from dual_model import DualModel, PLBartForConditionalGeneration
from utils import create_test_dataset, recover, dump_json, load_json
from metrics.bleu import corpus_bleu
from metrics.meteor import Meteor
from metrics.rouge import Rouge


def eval_accuracies(hypotheses, references, mode='dev'):
    assert (sorted(references.keys()) == sorted(hypotheses.keys()))

    # Compute BLEU scores
    _, bleu, _ = corpus_bleu(hypotheses, references)

    # Compute ROUGE scores
    rouge_calculator = Rouge()
    rouge_l, _ = rouge_calculator.compute_score(references, hypotheses)

    # Compute METEOR scores
    if mode == 'test':
        meteor_calculator = Meteor()
        meteor, _ = meteor_calculator.compute_score(references, hypotheses)
    else:
        meteor = 0

    return {
        'bleu': bleu * 100,
        'rouge_l': rouge_l * 100,
        'meteor': meteor * 100
    }


def valid_epoch(loader: Dataset,
                model: DualModel,
                num_beams: int,
                task: str,
                batch_size: int = 1,
                save_path: Union[str, None] = None
                ) -> dict:
    model.eval()

    hyp = {}
    ref = {}
    shown = [False for _ in range(10)]
    for ix in range(0, len(loader), batch_size):
        if ix + batch_size <= len(loader):
            raw_list = loader[ix: ix + batch_size]
        else:
            raw_list = loader[ix:]
        src_list = raw_list['src_list']
        tgt_list = raw_list['tgt_list']
        if not isinstance(tgt_list, list):
            tgt_list = [tgt_list]
        if task.lower() == "cs":
            hyps = model.cs_generate(src_list, num_beams)
        elif task.lower() == "cg":
            hyps = model.cg_generate(src_list, num_beams)
        else:
            raise RuntimeError("Only support task 'cs' and 'cg'. ")
        for i in range(len(hyps)):
            idx = len(hyp)
            hyp[idx] = [recover(hyps[i])]
            ref[idx] = [recover(tgt_list[i])]
        status = ix/len(loader) * 100
        status_idx = int(status // 10)
        if not shown[status_idx]:
            shown[status_idx] = True
            print(f"testing {round(status, 2)} %.")
    res = eval_accuracies(hyp, ref, mode='test')
    if save_path:
        results = [res]
        for i in range(len(hyp)):
            results.append({
                "ref": ref[i],
                "hyp": hyp[i]
            })
        dump_json(results, save_path)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",
                        type=str,
                        help="the id of checkpoint to test")
    parser.add_argument("--num_beams",
                        type=int,
                        help="Num-beams when generating")
    parser.add_argument("--use_dual", action="store_true")
    parser.add_argument("--code_lang", default="java", choices=['java', 'python'],
                        help="type of programming language")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str,
                        help="checkpoint dir, superior to save_dir + checkpoint")

    parser.add_argument("--lr", default=5e-5, type=float, help="learning rate")
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int)
    parser.add_argument('--epoches', default=100)
    parser.add_argument("--max_grad_norm", default=5.)
    parser.add_argument('--cp_steps', default=125000)

    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--test_only",
                        default=False,
                        type=bool,
                        help="only test")
    parser.add_argument("--model_name",
                        default="model",
                        type=str,
                        help="name of model")
    parser.add_argument("--save_dir",
                        default="java_results/",
                        type=str,
                        help="directory to save model and log")

    parser.add_argument("--test_code_path",
                        default="data/java/tmp/code.original")
    parser.add_argument("--test_summ_path",
                        default="data/java/tmp/javadoc.original")

    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        args.gpu = True
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
        args.checkpoint = os.path.split(args.checkpoint_dir)[-1].split('-')[-1]
        args.save_dir = os.path.split(args.checkpoint_dir)[0]
    else:
        checkpoint_dir = os.path.join(args.save_dir, f"checkpoint-{args.checkpoint}")
    model_state_dir = os.path.join(checkpoint_dir, "pytorch_model.bin")
    print(f"load model from {model_state_dir}")
    cs_model = PLBartForConditionalGeneration.from_pretrained(
        "./pretrained_cache/plbart-base")
    cs_tokenizer = PLBartTokenizer.from_pretrained("./pretrained_cache/plbart-base",
                                                   src_lang=args.code_lang,
                                                   tgt_lang="__en_XX__")
    cg_model = PLBartForConditionalGeneration.from_pretrained(
        "./pretrained_cache/plbart-base")
    cg_tokenizer = PLBartTokenizer.from_pretrained("./pretrained_cache/plbart-base",
                                                   src_lang="__en_XX__",
                                                   tgt_lang=args.code_lang)
    config_dict = load_json(os.path.join(checkpoint_dir, "config.json"))
    config = PretrainedConfig.from_dict(config_dict)
    dual_model = DualModel(config=config,
                           cs_model=cs_model,
                           cg_model=cg_model,
                           cs_tokenizer=cs_tokenizer,
                           cg_tokenizer=cs_tokenizer,
                           device=device)
    dual_model.load_state_dict(torch.load(model_state_dir))
    print("model loaded")

    cs_dataset = create_test_dataset(
        src_fp=args.test_code_path,
        tgt_fp=args.test_summ_path
    )
    cg_dataset = create_test_dataset(
        src_fp=args.test_summ_path,
        tgt_fp=args.test_code_path
    )

    print("validing CS task:")
    t = time.time()
    cs_eval = valid_epoch(loader=cs_dataset,
                          model=dual_model,
                          num_beams=args.num_beams,
                          task="cs",
                          batch_size=args.batch_size,
                          save_path=os.path.join(
                              args.save_dir, f"checkpoint-{args.checkpoint}-cs_test.json")
                          )
    print(f"Cost {time.time() - t} s")
    print(cs_eval)

    print("validing CG task:")
    t = time.time()
    cg_eval = valid_epoch(loader=cg_dataset,
                          model=dual_model,
                          num_beams=args.num_beams,
                          task="cg",
                          batch_size=args.batch_size,
                          save_path=os.path.join(
                              args.save_dir, f"checkpoint-{args.checkpoint}-cg_test.json"))
    print(f"Cost {time.time() - t} s")
    print(cg_eval)
