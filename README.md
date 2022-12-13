# NLP对抗训练

## 模型
+ Bert-Base-Chinese

## 数据集
+ ChnSentiCorp

## 方法

1. FGM
2. PGD
3. FreeLB

## 目录结构
+ 可直接运行 Adversarial.ipynb
+ config.py 为 参数
+ initializer.py 为 模型初始化代码
+ Runner.py 中 封装了 对于各种训练方法的Runner类
+ FGM.py PGD.py FreeLB.py 中为 插件式 对抗训练的代码
+ modelparams文件夹 为 用验证集保存模型训练过程中的参数
+ data 储存了使用的数据集


## 我正在做

+ 调试FreeLB参数以试图获得更好效果
+ 应用到更多数据集
+ 应用到NER任务