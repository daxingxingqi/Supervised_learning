# Supervised_learning
# 模型评估与选择
### 回归问题的评估：均方误差（mean squared error)
<div align=center><img width="350" src=resource/mean_squared_error.png></div>

### 分类问题的粗略评估：错误率与精度

**错误率**：
<div align=center><img width="450" src=resource/error_rate.png></div>

**Confusion Matrix**
<div align=center><img width="450" src=resource/confusion.png></div>

**Accuracy(准确率)**
<div align=center><img width="450" src=resource/accuracy.png></div>

NB:准确率不适用的情况-如下面，准确率达到99%，但是一个恶意邮件也没有捕捉到。
<div align=center><img width="450" src=resource/accuracy_bad.png></div>

### 分类任务的精确度量：查准率、查全率与Ｆ1
**Precision and Recall(精度和召回率 或者 查准率和查全率）**

精度：

- P=TP/(TP+FP)

- 对于诊断，我们并不需要很高的准确率，因为我们要避免的是sick的被诊断为Health（如图中红色叉子），如果Health被诊断为Sick则问题不大。
<div align=center><img width="450" src=resource/Precison_doc.png></div>

- 对于邮件，我们需要很高的准确率，因为我们要避免的是Not spam的被分类为Spam（如图中红色叉子）。
<div align=center><img width="450" src=resource/Precision_mail.png></div>

召回率：

- R=TP/(TP+FN)

- 对于诊断，我们并需要很高的召回率，因为我们要避免的是sick的被诊断为Health（如图中红色叉子）。
<div align=center><img width="450" src=resource/Recal_doc.png></div>

- 对于邮件，我们并不需要很高的召回率，因为我们要避免的是Not spam的被分类为Spam（如图中红色叉子）。
<div align=center><img width="450" src=resource/Recal_mail.png></div>

**F_beta**

- 其中β>0度量了查全率对查准率的相对重要性。

- β>1的时候查全率有更大影响；

- β<1的时候查准率有更大影响；

**ROC曲线（Receiver Operating Characteristic）**

ROC曲线主要关注TPrate和FPrate。

- True positive Rate = True Posivives/All Posivives 在所有positive中分对了多少
- False positive Rate = False Posivives/All Posivives 在所有negtive中分错了多少

> <div align=center><img width="450" src=resource/roc1.png></div>
> <div align=center><img width="450" src=resource/roc2.png></div>
> <div align=center><img width="450" src=resource/roc3.png></div>
> <div align=center><img width="450" src=resource/roc4.png></div>

**Precision-Recall 图**

<div align=center><img width="450" src=resource/pr.png></div>
如果一个学习器的PR曲线被另一个学习器的PR曲线完全包住，则后者的性能优于前者。上图中A明显优于C。如果两个学习器的PR曲线相交，就不能着急下结论了，这个时候一般回家算两条曲线下面包裹的面积，面积大的很明显取得PR双高的概率要更高。但是这个不太容易估算。对此，有人设计了BEP，平衡点（Break-Even Point）。
BEP是指查准率=查全率时候的取值，也就是画一条直线f(x)=x，曲线和这条直线的交点就是BEP。比如上图中A的BEP为0.8，B的BEP为0.72， 按照BEP来比较，学习器A优于B。但是，这也并不是绝对的，毕竟BEP还是过于简化了些。，更常用的方法是F1度量.

**K 折交叉验证** 

查看此链接-https://github.com/daxingxingqi/Supervised_learning/blob/master/model_eval_and_select.ipynb


**Learning curve**

<div align=center><img width="450" src=resource/learning_curve.png></div>

传统的机器学习算法（又被称为基于统计的机器学习）在数据量达到一定程度后，更多的数据无法提升模型的表现。深度学习的一个优势就是它可以把大量的数据利用起来，提升学习表现。

这里还有更多关于学习曲线的介绍：

https://www.coursera.org/learn/machine-learning/lecture/Kont7/learning-curves

http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
