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

---------------------------------------------------------------------------------------------------------------------------
# 线性回归 Linear Regression
NB：具体查看文档

机器学习是根据训练数据对变量之间的关系进行建模。当输出变量（相应变量） y∈ℝ 是连续值时，我们称之为`回归分析`。回归分析的意思就是用函数描述一个或多个预测变量与相应变量y之间的关系，并根据该模型预测新观测值对应的相应。

**回归模型的目目标函数：**

>机器器学习模型的目目标函数通常包含两项:损失函数 和正则项 ,分别度量量模型与训练数据的匹配程
度(损失函数越小小越匹配)和对模型复杂度的“惩罚”(避免过拟合)

<div align=center><img width="550" src=resource/linear_regression_objective_function.png></div>

**L2损失：**

<div align=center><img width="350" src=resource/mean_squared_error.png></div>

> 令残差r = y-f(x)表示模型预测值f(x)和真值y之间的差异，回归任务常用的损失函数是L2损失（least square）

<div align=center><img width="550" src=resource/linear_regression_l2_loss.png></div>

> 即残差的平方。所有样本的损失函数值被称为经验风险，表示模型与训练数据的拟合程度。当损失函数取L2损失时，所有样本的损失函数值之和称为残差平方和（residual sum of squares, RSS）:

<div align=center><img width="550" src=resource/linear_regression_l2_loss_sum.png></div>

> L2损失在回归分析中很常用。但是L2损失对于离群点（outliers）敏感。离群点通常远离大部分数据，如果根据大部分数据（去除离群点）得到理想模型，则残差r = y-f(x)（预测值f(x)和真值y)的绝对值比较大（训练数据中有离群点）。也就是说，算法会根据大部分数据得到一个理想模型，但是用损失函数调优时会把模型调大。因为残差变大。因此，采用L2损失而得到的理想模型对于离群点敏感。

**L1损失**

<div align=center><img width="450" src=resource/mean_absolute_error.png></div>

> 当数据中存在离群点的时候，可采用L1损失，即残差r=y-f(x)的绝对值；

<div align=center><img width="550" src=resource/linear_regression_l1_loss.png></div>

**L2正则:岭回归(Ridge Regression)**

<div align=center><img width="550" src=resource/Ridge_Regression.png></div>

其中 表示特征的维数, 为正则参数,控制正则惩罚的强度。

**L1正则:Lasso**

<div align=center><img width="550" src=resource/lasso.png></div>

其中 为正则参数,控制正则惩罚的强度。当λ取合适值时,Lasso)的结果是稀疏的(w的某些元素
系数为0),起到特征选择作用用。

**优化求解**

- 求解法

- 梯度下降法：思考以下情形：如果你的数据十分庞大，两种方法的计算速度都将会很缓慢。线性回归的最佳方式是将数据拆分成很多小批次。每个批次都大概具有相同数量的数据点。然后使用每个批次更新权重。这种方法叫做小批次梯度下降法。

优点：

实现简单，计算简单；

缺点：

不能拟合非线性数据；

----------------------------------------------------------------------------------------------------------------------------

# Logistic 回归
NB：具体查看文档

Logistic回归是⼀个⽤在分类任务的线性分类器。Logistic回归也是（深度）神经⽹络的基础，可以看作是只包含输⼊层和输出层的两层⽹络。同第⼀周课程中线性回归类似，我们也从经验⻛险最⼩、正则、优化、模型评估和模型选择等⽅⾯展开讨论。LR回归是在线性回归模型的基础上，使⽤sigmoid函数（logistic分布的累计分布函数），将线性模型的输出压缩到[0,1]之间，使其能表示概率。 LR本质仍然是⼀个线性模型，实现相对简单。在⼴告计算和推荐系统中使⽤频率极⾼，是点击率（CTR）预估模型的基本算法（在CTR部分会再介绍LR的在线学习⽅法）。同时，LR模型也是深度学习的基本组成单元（两层⽹络就是LR）。

LR回归属于概率性判别式模型。之所谓是概率性模型，是因为LR模型是有概率意义的（LR可以得到后验概率 ；⽽⾮概率模型如SVM，模型本身并没有概率意义）；之所以是判别式模型，是因为LR回归并没有对数据的分布进⾏建模，也就是说，LR模型并不知道数据的具体分布，⽽是直接将判别函数（这⾥是后验概率），或者说是分类超平⾯求解了出来。

> 判别式模型 VS 产生式模型：
> - 产生式模型通过求解p(y =c|x),来实现分类。
> - 判别式模型通过假设p(y=1|x)=delta(W^Tx)，来实现分类

优点：

　1、实现简单；

　2、分类时计算量非常小，速度很快，存储资源低；

缺点：

　　1、容易欠拟合，一般准确度不太高

　　2、只能处理两分类问题（在此基础上衍生出来的softmax可以用于多分类），且必须线性可分；
  
-----------------------------------------------------------------------------------------------------------------------------
 # 决策树
 
信息熵的计算公式如下:
<div align=center><img width="550" src=resource/tree.png></div>

优点：

计算量简单，可解释性强，比较适合处理有缺失属性值的样本，能够处理不相关的特征；

缺点：

容易过拟合（后续出现了随机森林，减小了过拟合现象）；
 
