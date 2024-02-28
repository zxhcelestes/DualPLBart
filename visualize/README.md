# 可视化模块
在此文件夹中加入前端可视化代码，外型可以参考谷歌翻译
![](figures/google-trans.png)
代码可选项为：Java/Python  
自然语言可选项为：English（En）

目前Java-En的模型已基本开发完成，见`results/models`，模型加载的方式参考`test.py`。Python-En的模型结构与Java-En相同，要求在代码中设置目标模型的路径，通过调整路径即可选择需要加载的模型。

最终要求执行`python DUALPLBART/run.py`或点击`.exe`文件即可运行程序。
