# Supervised_learning
# 模型评估与选择
## 评估
### 回归问题的评估：均方误差（mean squared error)
<div align=center><img width="350" src=resource/mean_squared_error.png></div>

### 分类问题的粗略评估：错误率与精度
错误率是分类错误的样本数占样本总数的比例，精度则是分类正确的样本数占总数的比例。

错误率可以定义为：
<div align=center><img width="550" src=resource/error_rate.png></div>
精度可以定义为：
<div align=center><img width="550" src=resource/accuracy2.png></div>


**Confusion Matrix**
<div align=center><img width="550" src=resource/confusion.png></div>

**Accuracy(准确率)**
<div align=center><img width="550" src=resource/accuracy.png></div>

NB:准确率不适用的情况-如下面，准确率达到99%，但是一个恶意邮件也没有捕捉到。
<div align=center><img width="550" src=resource/accuracy_bad.png></div>

### 分类任务的精确度量：查准率、查全率与Ｆ1
**Precision and Recall(精度和召回率或者查准率和查全率）**

精度：

P=TP/TP+FP
- 对于诊断，我们并不需要很高的准确率，因为我们要避免的是sick的被诊断为Health（如图中红色叉子），如果Health被诊断为Sick则问题不大。
<div align=center><img width="550" src=resource/Precison_doc.png></div>

- 对于邮件，我们需要很高的准确率，因为我们要避免的是Not spam的被分类为Spam（如图中红色叉子）。
<div align=center><img width="550" src=resource/Precision_mail.png></div>

召回率：

R=TP/TP+FN
- 对于诊断，我们并需要很高的召回率，因为我们要避免的是sick的被诊断为Health（如图中红色叉子）。
<div align=center><img width="550" src=resource/Recal_doc.png></div>

- 对于邮件，我们并不需要很高的召回率，因为我们要避免的是Not spam的被分类为Spam（如图中红色叉子）。
<div align=center><img width="550" src=resource/Recal_mail.png></div>

**经验误差与过拟合**

错误率（error rate）：我们通常把分类错误的样本数占样本总数的比例称为错误率。

精度（accuracy）：精度等与１减去错误率。

误差（error）：我们把机器学习的实际预测输出与样本的真实输出之间的差异称之为误差。

训练误差（training error)：机器学习在训练集上的误差称之为训练误差。

泛化误差（generaliztion error）：机器学习在新样本上的误差称之为泛化误差。

>很明显，我们希望得到泛化误差很小的学习器，但是由于不知道新样本是什么样的，所以只能尽量让训练误差最小。看上去这没什么问题，但是事实上凡事都需要有一个度，很多时候我们甚至能得到在训练集中分类错误率为０，精度为100%的学习器，但是大多数时候这样的模型并不好用。当学习器把训练样本学的太好了，很可能会把训练样本本身的特点当成了所有潜在样本的特点，事实上这些特点是训练样本所独有的，那么最终的泛化误差就会很大了，也就是说返回性能下降了，这种现象在机器学习中称作过拟合。与过拟合相对的是欠拟合，简单的来说，就是对训练样本的学习还不太到位。回到最开始的问题，很明显要评估一个模型是否适合解决此类问题最好的办法就是利用泛化误差，但是新样本到底是个什么鬼我们终究还是不知道的，也就无法得到泛化误差，而使用训练误差又会存在过拟合的问题，那么到底应该怎么去评估呢?

