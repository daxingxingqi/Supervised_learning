# Supervised_learning
# 模型评估与选择
## 评估
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

**ROC曲线（Receiver Operating Characteristic）
- <div align=center><img width="450" src=resource/roc1.png></div>
- <div align=center><img width="450" src=resource/roc2.png></div>
- <div align=center><img width="450" src=resource/roc3.png></div>
- <div align=center><img width="450" src=resource/roc4.png></div>
