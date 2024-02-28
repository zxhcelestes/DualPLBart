python -u test.py \
--checkpoint 159756 \
--num_beams 1 \
--code_lang java \
--gpu \
--model_name sep_model \
--save_dir result_models/sep_models \
--test_code_path data/dual_data/java/test/code.original \
--test_summ_path data/dual_data/java/test/javadoc.original
