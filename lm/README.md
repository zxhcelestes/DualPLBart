# 语言模型
基于`https://github.com/Bolin0215/CSCGDual/tree/master`改进
语言模型用于计算代码和注释的先验概率

```bash
python lm/main.py --data data/lm_data/java --save result_models/lm_models/lm_java.pt --lang __java__ --batch_size 32 --cuda --tied
```
