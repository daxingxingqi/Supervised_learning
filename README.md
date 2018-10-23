# Supervised_learning
# 模型评估与选择
## 评估指标

**Confusion Matrix**
<div align=center><img width="550" src=resource/confusion.png></div>

**Accuracy(准确率)**
<div align=center><img width="550" src=resource/accuracy.png></div>

NB:准确率不适用的情况-如下面，准确率达到99%，但是一个恶意邮件也没有捕捉到。
<div align=center><img width="550" src=resource/accuracy_bad.png></div>

**Precision and Recall(精度和召回率或者查准率和查全率）**

精度：

- 对于诊断，我们并不需要很高的准确率，因为我们要避免的是sick的被诊断为Health（如图中红色叉子），如果Health被诊断为Sick则问题不大。
<div align=center><img width="550" src=resource/Precison_doc.png></div>

- 对于邮件，我们需要很高的准确率，因为我们要避免的是Not spam的被分类为Spam（如图中红色叉子）。
<div align=center><img width="550" src=resource/Precision_mail.png></div>

召回率：

- 对于诊断，我们并需要很高的召回率，因为我们要避免的是sick的被诊断为Health（如图中红色叉子）。
<div align=center><img width="550" src=resource/Recal_doc.png></div>

- 对于邮件，我们并不需要很高的召回率，因为我们要避免的是Not spam的被分类为Spam（如图中红色叉子）。
<div align=center><img width="550" src=resource/Recal_mail.png></div>
