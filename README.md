# Adversarial Training for Chinese NLP Tasks
*Your name and Student ID: [李培基], [20307140044]*

## 任务介绍
对抗训练(Adversarial Training)是一种"防守"策略，在训练过程中对模型进行attack,从而提升模型的鲁棒性，以试图在测试过程中能够取得更好结果。
对抗训练最初被应用于CV任务中，然而实验证明，其在NLP任务中往往可以取得更好的结果。在NLP领域，常用的对抗训练方法即为对word_embedding层进行扰动，虽然这种方法并不像图像领域的attack一样能被很好的"解释"，因为扰动后的word vector似乎并不能够找到相应的word映射，但是以往的实验证明其确实能够提升模型在测试集上的准确率。
我正在做的是事情是，尝试学习并使用当下几种主流的对抗训练方法(FGM,PGD,FreeLB),并将其应用于使用了预训练模型的**中文**NLP任务的训练过程中。
我们对比不使用对抗训练，与使用了不同对抗训练的方法，在经过同样轮次训练后，模型所达到的效果。

## 应用到 NLP Tasks
+ Chinese SSC 中文单句情感分类
+ Chinese NER 中文命名实体识别
## 预训练模型
+ Bert-Base-Chinese(Huggingface)

## 使用的模型结构
+ Bert + Linear
+ Bert + CRF
## 数据集
+ ChnSentiCorp
+ peoples_daily_ner
## 使用的对抗训练方法

+ FGM
+ PGD
+ FreeLB

## 目录结构
+ SSC.ipynb 与 NER.ipynb 为两个中文NLP任务包含对抗训练方法的运行脚本
+ report中有Final PJ的论文报告。
+ config.py是一些参数, initializer.py是一些模型初始化代码和类、函数的定义
+ Runner.py 中封装了对于各种训练方法的Runner类
+ FGM.py PGD.py FreeLB.py 中为 插件式 对抗训练的代码实现
+ modelparams文件夹为验证集所的保存模型训练过程中的参数
+ data 储存了使用的数据集

config.py Runner.py 和 initializer.py 都是对于SSC任务所专门写的文件，后来发现专门写出来必要性不大，对于NER任务，全部类和函数的代码都写在了 NER.ipynb 脚本中

## Still Working on...

+ 调试FreeLB参数以试图获得更好效果
+ 应用到更多数据集
+ 应用到其他NLP任务中

## 感谢
此任务在颜航博士的指导下进行，他指导我去阅读了相应的论文并帮我指出了许多问题。
感谢邱老师，颜航博士和李鹏学长的帮助！
感谢实验室为我的任务提供了充足的算力！！！