# DualPLBART

## 安装环境
```bash
conda create --name dualplbart python=3.9
conda activate dualpart
pip install -e .
```
建议先自行安装好torch。如有新的依赖库，安装后在`setup.py`文件`install_requires`列表添加。

## 单独训练
```bash
python train_sep.py --code_lang java --gpu --model_name sep_model --save_dir result_models/sep_models --train_code_path data/dual_data/java/tmp3/code.original --train_summ_path data/dual_data/java/tmp3/javadoc.original

pyd train_sep.py --code_lang java --gpu --model_name sep_model --save_dir result_models/sep_models --train_code_path data/dual_data/java/tmp3/code.original --train_summ_path data/dual_data/java/tmp3/javadoc.original
```

## 对偶训练
```bash
python train_dual.py --use_dual --use_attn --code_lang java --gpu --model_name dual_model --save_dir result_models/dual_models2 --train_code_path data/dual_data/java/train/code.original --train_summ_path data/dual_data/java/train/javadoc.original --train_code_score_path data/dual_data/java/train/code.score --train_summ_score_path data/dual_data/java/train/javadoc.score --lambda_dual 0.001 --lambda_attn 0.001 --reuse

pyd train_dual.py --use_dual --use_attn --code_lang java --gpu --model_name dual_model --save_dir result_models/dual_models2 --train_code_path data/dual_data/java/train/code.original --train_summ_path data/dual_data/java/train/javadoc.original --train_code_score_path data/dual_data/java/train/code.score --train_summ_score_path data/dual_data/java/train/javadoc.score --lambda_dual 0.001 --lambda_attn 0.001 --reuse
```

## 测试模型
```bash
python -u test.py --checkpoint_dir $dir --num_beams 1 --code_lang java --gpu --model_name dual_model --save_dir result_models/dual_models --test_code_path data/dual_data/java/test/code.original --test_summ_path data/dual_data/java/test/javadoc.original --batch_size 16

pyd -u test.py --checkpoint_dir $dir --num_beams 1 --code_lang java --gpu --model_name dual_model --save_dir result_models/dual_models --test_code_path data/dual_data/java/test/code.original --test_summ_path data/dual_data/java/test/javadoc.original --batch_size 16
```
