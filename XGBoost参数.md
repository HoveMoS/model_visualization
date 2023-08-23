# XGBoost参数





# SKLearn-API

XGBoost使用**key-value**字典的方式存储参数：

```
#部分重要参数
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',  # 多分类的问题
    'num_class': 10,               # 类别数，与 multisoftmax 并用
    'gamma': 0.1,                  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth': 12,               # 构建树的深度，越大越容易过拟合
    'lambda': 2,                   # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample': 0.7,              # 随机采样训练样本
    'colsample_bytree': 0.7,       # 生成树时进行的列采样
    'min_child_weight': 3,
    'silent': 1,                   # 设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.007,                  # 如同学习率
    'seed': 1000,
    'nthread': 4,                  # cpu 线程数
}
```





## XGBClassifier

```
from xgboost.sklearn import XGBClassifier
 
clf = XGBClassifier(
    silent=0,  # 设置成1则没有运行信息输出，最好是设置为0，是否在运行升级时打印消息
    # nthread = 4  # CPU 线程数 默认最大
    learning_rate=0.3 , # 如同学习率
    min_child_weight = 1,
    # 这个参数默认为1，是每个叶子里面h的和至少是多少，对正负样本不均衡时的0-1分类而言
    # 假设h在0.01附近，min_child_weight为1 意味着叶子节点中最少需要包含100个样本
    # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
    max_depth=6, # 构建树的深度，越大越容易过拟合
    gamma = 0,# 树的叶子节点上做进一步分区所需的最小损失减少，越大越保守，一般0.1 0.2这样子
    subsample=1, # 随机采样训练样本，训练实例的子采样比
    max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计
    colsample_bytree=1, # 生成树时进行的列采样
    reg_lambda=1, #控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合
    # reg_alpha=0, # L1正则项参数
    # scale_pos_weight =1 # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛，平衡正负权重
    # objective = 'multi:softmax', # 多分类问题，指定学习任务和响应的学习目标
    # num_class = 10,  # 类别数，多分类与multisoftmax并用
    n_estimators=100,  # 树的个数
    seed = 1000,  # 随机种子
    # eval_metric ='auc'
)
```







**xgboost完整参数解析**

![图片](https://mmbiz.qpic.cn/mmbiz_png/njjfaJS7c9pxRm5a77PVibrWqJOLAqhdWWwIwBsdGxcpl25erhcWkO8zKSsvLib6HVfYHQTmPz43jsO5jkEibqo5Q/640?wx_fmt=png&wxfrom=5&wx_lazy=1&wx_co=1)





## xgboost.XGBRegressor

xgboost 给出了针对scikit-learn 接口的API。

**xgboost.XGBRegressor： 它实现了scikit-learn 的回归模型API**

```
class xgboost.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100, 
     silent=True, objective='reg:linear', booster='gbtree', n_jobs=1, nthread=None, 
     gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
     colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, 
     base_score=0.5, random_state=0, seed=None, missing=None, **kwargs)
    
```



参数：

- max_depth： 一个整数，表示子树的最大深度
- learning_rate： 一个浮点数，表示学习率
- n_estimators：一个整数，也就是弱学习器的最大迭代次数，或者说最大的弱学习器的个数。
- silent： 一个布尔值。如果为False，则打印中间信息
- objective： 一个字符串或者可调用对象，指定了目标函数。其函数签名为：objective(y_true,y_pred) -> gra,hess。 其中：
  - y_true： 一个形状为[n_sample] 的序列，表示真实的标签值
  - y_pred： 一个形状为[n_sample] 的序列，表示预测的标签值
  - grad： 一个形状为[n_sample] 的序列，表示每个样本处的梯度
  - hess： 一个形状为[n_sample] 的序列，表示每个样本处的二阶偏导数
- booster： 一个字符串。指定了用哪一种基模型。可以为：’gbtree’,’gblinear’,’dart’
- n_jobs： 一个整数，指定了并行度，即开启多少个线程来训练。如果为-1，则使用所有的CPU
- gamma： 一个浮点数，也称作最小划分损失min_split_loss。 它刻画的是：对于一个叶子节点，当对它采取划分之后，损失函数的降低值的阈值。
- min_child_weight： 一个整数，子节点的权重阈值。它刻画的是：对于一个叶子节点，当对它采取划分之后，它的所有子节点的权重之和的阈值。
- max_delta_step： 一个整数，每棵树的权重估计时的最大delta step。取值范围为，0 表示没有限制，默认值为 0 。
- subsample：一个浮点数，对训练样本的采样比例。取值范围为 (0,1]，默认值为 1 。如果为5， 表示随机使用一半的训练样本来训练子树。它有助于缓解过拟合。
- colsample_bytree： 一个浮点数，构建子树时，对特征的采样比例。取值范围为 (0,1]， 默认值为 1。如果为5， 表示随机使用一半的特征来训练子树。它有助于缓解过拟合。
- colsample_bylevel： 一个浮点数，寻找划分点时，对特征的采样比例。取值范围为 (0,1]， 默认值为 1。如果为5， 表示随机使用一半的特征来寻找最佳划分点。它有助于缓解过拟合。
- reg_alpha： 一个浮点数，是L1 正则化系数。它是xgb 的alpha 参数
- reg_lambda： 一个浮点数，是L2 正则化系数。它是xgb 的lambda 参数
- scale_pos_weight： 一个浮点数，用于调整正负样本的权重，常用于类别不平衡的分类问题。默认为 1。
- base_score：一个浮点数， 给所有样本的一个初始的预测得分。它引入了全局的bias
- random_state： 一个整数，表示随机数种子。
- missing： 一个浮点数，它的值代表发生了数据缺失。默认为nan
- kwargs： 一个字典，给出了关键字参数。它用于设置Booster 对象





## xgboost.XGBClassifier

**xgboost.XGBClassifier ：它实现了scikit-learn 的分类模型API**

```
class xgboost.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, 
     silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1,
     nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1,
     colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1,
     scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, 
     missing=None, **kwargs)
```



参数参考xgboost.XGBRegressor

**xgboost.XGBClassifier 和 xgboost.XGBRegressor 的方法：**

- fit(X, y, sample_weight=None, eval_set=None, eval_metric=None,early_stopping_rounds=None,verbose=True, xgb_model=None) 训练模型
  - X： 一个array-like，表示训练集
  - y： 一个序列，表示标记
  - sample_weight： 一个序列，给出了每个样本的权重
  - eval_set： 一个列表，元素为(X,y)，给出了验证集及其标签。它们用于早停。如果有多个验证集，则使用最后一个
  - eval_metric： 一个字符串或者可调用对象，用于evaluation metric。如果为字符串，则是内置的度量函数的名字；如果为可调用对象，则它的签名为(y_pred,y_true)==>(str,value)
  - early_stopping_rounds： 指定早停的次数。参考train()
  - verbose： 一个布尔值。如果为True，则打印验证集的评估结果。
  - xgb_model：一个Booster实例，或者一个存储了xgboost 模型的文件的文件名。它给出了待训练的模型。这种做法允许连续训练。
- predict(data, output_margin=False, ntree_limit=0) 执行预测
  - data： 一个 DMatrix 对象，表示测试集
  - output_margin： 一个布尔值。表示是否输出原始的、未经过转换的margin value
  - ntree_limit： 一个整数。表示使用多少棵子树来预测。默认值为0，表示使用所有的子树。
  - 如果训练的时候发生了早停，则你可以使用best_ntree_limit。
- .predict_proba(data, output_margin=False, ntree_limit=0) 执行预测，预测的是各类别的概率。它只用于分类问题，返回的是预测各类别的概率。参数：参考.predict()
- .evals_result()： 返回一个字典，给出了各个验证集在各个验证参数上的历史值。它不同于cv() 函数的返回值。cv() 函数返回evaluation history 是早停时刻的。而这里返回的是所有的历史值













# XGBoost-API-官网



https://xgboost.readthedocs.io/en/latest/parameter.html





在运行XGBoost之前，我们必须设置三类参数：通用参数、Booster参数和任务参数。

- **一般参数**与我们使用哪个助推器进行助推有关，通常是树模型或线性模型
- **助推器参数**取决于您选择的助推器
- **学习任务参数**决定学习场景。例如，回归任务可以使用与排名任务不同的参数。
- **命令行参数**与 XGBoost CLI 版本的行为相关。





- [全局配置](https://xgboost.readthedocs.io/en/latest/parameter.html#global-configuration)
- [一般参数](https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters)
- 树助推器的参数
  - [树助推器的参数](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tree-booster)
  - [分类特征的参数](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-categorical-feature)
  - [Dart Booster 的附加参数 ( `booster=dart`)](https://xgboost.readthedocs.io/en/latest/parameter.html#additional-parameters-for-dart-booster-booster-dart)
  - [线性助推器的参数 ( `booster=gblinear`)](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-linear-booster-booster-gblinear)
- [学习任务参数](https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters)
  - [Tweedie 回归的参数 ( `objective=reg:tweedie`)](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tweedie-regression-objective-reg-tweedie)
  - [使用 Pseudo-Huber 的参数 ( `reg:pseudohubererror`)](https://xgboost.readthedocs.io/en/latest/parameter.html#parameter-for-using-pseudo-huber-reg-pseudohubererror)
  - [使用分位数损失的参数 ( `reg:quantileerror`)](https://xgboost.readthedocs.io/en/latest/parameter.html#parameter-for-using-quantile-loss-reg-quantileerror)
  - [使用 AFT 生存损失 ( `survival:aft`) 和 AFT 度量的负对数似然 ( `aft-nloglik`)的参数](https://xgboost.readthedocs.io/en/latest/parameter.html#parameter-for-using-aft-survival-loss-survival-aft-and-negative-log-likelihood-of-aft-metric-aft-nloglik)
  - [用于学习排名的参数 ( `rank:ndcg`, `rank:map`, `rank:pairwise`)](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-learning-to-rank-rank-ndcg-rank-map-rank-pairwise)
- [命令行参数](https://xgboost.readthedocs.io/en/latest/parameter.html#command-line-parameters)



## 全局配置

[`xgboost.config_context()`](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.config_context)可以使用(Python) 或(R)在全局范围内设置以下参数`xgb.set.config()`。

- `verbosity`：打印消息的详细程度。有效值为 0（静默）、1（警告）、2（信息）和 3（调试）。
- `use_rmm`：是否使用 RAPIDS Memory Manager (RMM) 来分配 GPU 内存。此选项仅适用于在启用 RMM 插件的情况下构建（编译）XGBoost 的情况。有效值为`true`和`false`。



## 一般参数

- `booster`[默认= `gbtree`]

  - 使用哪种助推器。可以是`gbtree`,`gblinear`或`dart`; `gbtree`并`dart`使用基于树的模型，同时`gblinear`使用线性函数。

- `verbosity`[默认=1]

  - 打印消息的详细程度。有效值为 0（静默）、1（警告）、2（信息）、3（调试）。有时，XGBoost 尝试根据启发式更改配置，这会显示为警告消息。如果出现意外行为，请尝试增加详细程度的值。

- `validate_parameters`[默认为`false`，Python、R 和 CLI 界面除外]

  - 当设置为True时，XGBoost将对输入参数进行验证以检查参数是否被使用。

- `nthread`[如果未设置，则默认为最大可用线程数]

  - 用于运行 XGBoost 的并行线程数。选择它时，请记住线程争用和超线程。
  - NET补充：如果你希望以最大速度运行，建议不设置这个参数，模型将自动获得最大线程

- `disable_default_eval_metric`[默认= `false`]

  - 标记以禁用默认指标。设置为 1 或`true`禁用。

- `num_feature`[由XGBoost自动设置，无需用户设置]

  - boosting中使用的特征维度，设置为特征的最大维度

    



## 助推器的参数

### 树助推器的参数

- `eta`[默认=0.3，别名：`learning_rate`]

  - 更新中使用步长收缩来防止过度拟合。在每个boosting步骤之后，我们可以直接得到新特征的权重，并`eta`缩小特征权重以使boosting过程更加保守。
  - 范围：[0,1]
  - NET补充：
    - 为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。 eta通过缩减特征的权重使提升计算过程更加保守。`缺省值为0.3`
    - 通常最后设置eta为0.01~0.2

- `gamma`[默认=0，别名：`min_split_loss`]

  - 在树的叶节点上进行进一步划分所需的最小损失减少。越大`gamma`，算法越保守。
  - 范围：[0,无穷大]
  - NET补充：
    - 模型在默认情况下，对于一个节点的划分只有在其loss function 得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值
    - gamma值使得算法更conservation，且其值依赖于loss function ，在模型中应该进行调参。

- `max_depth`[默认=6]

  - 树的最大深度。增加该值将使模型更加复杂并且更容易过度拟合。0 表示深度没有限制。请注意，XGBoost 在训练深度树时会大量消耗内存。`exact`树方法需要非零值。
  - 范围：[0,无穷大]
  - NET补充：
    - 指树的最大深度
    - 树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合
    - 建议通过交叉验证（xgb.cv ) 进行调参
    - 通常取值：3-10

- `min_child_weight`[默认=1]

  - 子级所需的实例权重（粗麻布）的最小总和。如果树分区步骤产生的叶节点的实例权重之和小于`min_child_weight`，则构建过程将放弃进一步分区。在线性回归任务中，这仅对应于每个节点中所需的最小实例数。越大`min_child_weight`，算法越保守。
  - 范围：[0,无穷大]
  - NET补充：孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。

- `max_delta_step`[默认=0]

  - 我们允许每个叶子输出的最大增量步长。如果该值设置为0，则表示没有约束。如果将其设置为正值，则可以帮助使更新步骤更加保守。通常不需要此参数，但当类别极度不平衡时，它可能有助于逻辑回归。将其设置为 1-10 的值可能有助于控制更新。
  - 范围：[0,无穷大]
  - NET补充：
  - - 如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    - 通常不需要设置这个值，但在使用logistics 回归时，若类别极度不平衡，则调整该参数可能有效果

- `subsample`[默认=1]

  - 训练实例的子样本比率。将其设置为 0.5 意味着 XGBoost 将在生长树之前随机采样一半的训练数据。这将防止过度拟合。子采样将在每次提升迭代中发生一次。
  - 范围：(0,1]
  - NET补充：用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。

- `sampling_method`[默认= `uniform`]

  - 用于对训练实例进行采样的方法。
  - `uniform`：每个训练实例被选择的概率相等。通常设置 `subsample`>= 0.5 以获得良好结果。
  - `gradient_based`：每个训练实例的选择概率与 梯度的*正则化绝对值*成正比（更具体地说，g2+λh2）。 `subsample`可以设置为低至 0.1，而不会损失模型精度。`tree_method`请注意，仅当设置为 时才支持此采样方法`gpu_hist`；其他树方法仅支持`uniform`采样。

- `colsample_bytree`, `colsample_bylevel`, `colsample_bynode`[默认=1]

  - 这是用于对列进行二次采样的一系列参数。

  - 所有`colsample_by*`参数的范围均为 (0, 1]，默认值为 1，并指定要进行二次采样的列的分数。

  - `colsample_bytree`是构建每棵树时列的子样本比率。每构建一棵树就会进行一次子采样。

  - `colsample_bylevel`是每个级别的列的子样本比率。树中每达到一个新的深度级别，就会进行一次子采样。列是从为当前树选择的列集中进行二次采样的。

  - `colsample_bynode`是每个节点（分割）的列的子样本比率。每次评估新的分割时都会进行二次采样。列是从为当前级别选择的列集进行二次采样的。

  - `colsample_by*`参数累积作用。例如，64 个特征的组合将在每次分割时留下 8 个特征可供选择。`{'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5}`

    使用Python或R包，可以设置`feature_weights`DMatrix来定义使用列采样时选择每个特征的概率。sklearn 接口中的方法有一个类似的参数`fit`。

    

    NET补充：

    - colsample_bytree在建立树时对特征随机采样的比例。`缺省值为1`

    - colsample_bylevel[default=1]

      - 决定每次节点划分时子样例的比例
      - 通常不使用，因为subsample和colsample_bytree已经可以起到相同的作用了

      

- `lambda`[默认=1，别名：`reg_lambda`]

  - 权重的 L2 正则化项。增加该值将使模型更加保守。
  - 范围：[0,∞]

- `alpha`[默认=0，别名：`reg_alpha`]

  - 权重的 L1 正则化项。增加该值将使模型更加保守。
  - 范围：[0,∞]

- `tree_method`字符串 [默认= `auto`]

  - XGBoost 中使用的树构建算法。[请参阅参考文件](http://arxiv.org/abs/1603.02754)和[树方法](https://xgboost.readthedocs.io/en/latest/treemethod.html)中的描述。
  - XGBoost 支持 `approx`、`hist`和`gpu_hist`用于分布式训练。对外部存储器的实验支持可用于`approx`和`gpu_hist`。
  - 选项：`auto`、`exact`、`approx`、`hist`、`gpu_hist`，这是常用更新程序的组合。对于其他更新程序，例如`refresh`，直接设置参数`updater`。
    - `auto`：使用启发式选择最快的方法。
      - 对于小数据集，`exact`将使用精确贪婪（ ）。
      - 对于较大的数据集，`approx`将选择近似算法（ ）。建议尝试使用大型数据集`hist`来`gpu_hist`获得更高的性能。( `gpu_hist`) 支持.`external memory`
      - 因为旧的行为在单机中总是使用精确贪婪，所以当选择近似算法时用户将收到一条消息来通知此选择。
    - `exact`：精确的贪心算法。枚举所有拆分候选者。
    - `approx`：使用分位数草图和梯度直方图的近似贪婪算法。
    - `hist`：更快的直方图优化近似贪婪算法。
    - `gpu_hist`：算法的GPU实现`hist`。

- `scale_pos_weight`[默认=1]

  - 控制正负权重的平衡，对于不平衡的类很有用。要考虑的典型值：。有关更多讨论，请参阅[参数调整。](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)另请参阅 Higgs Kaggle 竞赛演示，了解示例：[R](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-train.R)、[py1](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-numpy.py)、[py2](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-cv.py)、[py3](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py)。`sum(negative instances) / sum(positive instances)`
  - NET补充：大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛

- `updater`

  - 逗号分隔的字符串，定义要运行的树更新程序的序列，提供构建和修改树的模块化方法。这是一个高级参数，通常根据一些其他参数自动设置。然而，它也可以由用户显式设置。存在以下更新程序：
    - `grow_colmaker`：非分布式基于列的树构建。
    - `grow_histmaker`：基于直方图计数的全局提议的基于行的数据分割的分布式树构建。
    - `grow_quantile_histmaker`：使用量化直方图生长树。
    - `grow_gpu_hist`：使用 GPU 种植树。
    - `sync`：同步所有分布式节点中的树。
    - `refresh`：根据当前数据刷新树的统计数据和/或叶子值。请注意，不执行数据行的随机子采样。
    - `prune`：修剪 loss < min_split_loss （或 gamma）的分割以及深度大于 的节点`max_depth`。

- `refresh_leaf`[默认=1]

  - 这是更新器的一个参数`refresh`。当此标志为 1 时，树叶和树节点的统计信息都会更新。当它为 0 时，仅更新节点统计信息。

- `process_type`[默认= `default`]

  - 一种要运行的提升过程。
  - 选择：`default`,`update`
    - `default`：创建新树的正常提升过程。
    - `update`：从现有模型开始，仅更新其树。在每次提升迭代中，都会从初始模型中获取一棵树，为该树运行指定的更新程序序列，并将修改后的树添加到新模型中。新模型将具有相同或更少数量的树，具体取决于执行的提升迭代次数。目前，以下内置更新程序可以有意义地与此进程类型一起使用：`refresh`、`prune`。使用 时`process_type=update`，无法使用创建新树的更新程序。

- `grow_policy`[默认= `depthwise`]

  - 控制将新节点添加到树的方式。
  - 目前仅当`tree_method`设置为`hist`,`approx`或时才支持`gpu_hist`。
  - 选择：`depthwise`,`lossguide`

    - `depthwise`：在最接近根的节点处分割。
    - `lossguide`：在损失变化最大的节点处进行分割。
  
- `max_leaves`[默认=0]

  - 要添加的最大节点数。不被树方法使用`exact`。

- `max_bin`，[默认=256]

  - 仅当`tree_method`设置为`hist`、`approx`或 时才使用`gpu_hist`。
  - 用于存储连续特征的离散箱的最大数量。
  - 增加这个数字可以提高分割的最优性，但代价是增加计算时间。

- `predictor`, [默认= `auto`]

  - 要使用的预测器算法的类型。提供相同的结果，但允许使用 GPU 或 CPU。
    - `auto`：根据启发式配置预测器。
    - `cpu_predictor`：多核CPU预测算法。
    - `gpu_predictor`：使用 GPU 进行预测。`tree_method`当是时使用`gpu_hist`。当`predictor`设置为默认值时`auto`，`gpu_hist`树方法能够提供基于 GPU 的预测，而无需将训练数据复制到 GPU 内存。如果`gpu_predictor`明确指定，则所有数据都会复制到 GPU 中，仅建议用于执行预测任务。

- `num_parallel_tree`，[默认=1]

  - 每次迭代期间构建的并行树的数量。此选项用于支持增强随机森林。

- `monotone_constraints`

  - 变量单调性的约束。有关详细信息，请参阅[单调约束。](https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html)

- `interaction_constraints`

  - 代表允许交互的交互约束。约束必须以嵌套列表的形式指定，例如，其中每个内部列表是一组允许彼此交互的特征索引。有关详细信息，请参阅[功能交互约束。](https://xgboost.readthedocs.io/en/latest/tutorials/feature_interaction_constraint.html)`[[0, 1], [2, 3, 4]]`

- `multi_strategy`, [默认 = `one_output_per_tree`]

  *2.0.0 版本中的新增功能。*

  笔记

  该参数正在处理中。

  - 该策略用于训练多目标模型，包括多目标回归和多类分类。有关详细信息，请参阅[多个输出](https://xgboost.readthedocs.io/en/latest/tutorials/multioutput.html)。
    - `one_output_per_tree`：每个目标一个模型。
    - `multi_output_tree`：使用多目标树。



### 分类特征的参数

这些参数仅用于分类数据的训练。有关详细信息，请参阅 [分类数据。](https://xgboost.readthedocs.io/en/latest/tutorials/categorical.html)

笔记

这些参数是实验性的。`exact`尚不支持树方法。

- `max_cat_to_onehot`

  *1.6.0 版本中的新增功能。*

  - 用于决定 XGBoost 是否应该对分类数据使用基于单热编码的分割的阈值。当类别数量小于阈值时，选择 one-hot 编码，否则类别将被划分为子节点。

- `max_cat_threshold`

  *1.7.0 版本中的新增功能。*

  - 每次分割考虑的最大类别数。仅用于基于分区的分割，以防止过度拟合。

### Dart Booster 的附加参数 ( `booster=dart`)[](https://xgboost.readthedocs.io/en/latest/parameter.html#additional-parameters-for-dart-booster-booster-dart)

笔记

`predict()`与 DART booster 一起使用

如果助推器对象是 DART 类型，`predict()`将执行 dropouts，即仅评估某些树。`data`如果不是训练数据，这将产生不正确的结果。要在测试集上获得正确的结果，请设置`iteration_range`为非零值，例如

```
preds = bst.predict(dtest, iteration_range=(0, num_round))
```

- `sample_type`[默认= `uniform`]
  - 采样算法的类型。
    - `uniform`：统一选择掉落的树木。
    - `weighted`：按照重量比例选择掉落的树木。
- `normalize_type`[默认= `tree`]
  - 归一化算法的类型。
    - `tree`：新树的重量与每棵掉落的树的重量相同。
      - 新树的重量是。`1 / (k + learning_rate)`
      - 掉落的树木按 因子缩放。`k / (k + learning_rate)`
    - `forest`：新树的权重与掉落的树（森林）的总和相同。
      - 新树的重量是。`1 / (1 + learning_rate)`
      - 掉落的树木按 因子缩放。`1 / (1 + learning_rate)`
- `rate_drop`[默认=0.0]
  - Dropout 率（在 Dropout 期间掉落的先前树的一小部分）。
  - 范围：[0.0，1.0]
- `one_drop`[默认=0]
  - 启用此标志后，在 dropout 期间始终会删除至少一棵树（允许原始 DART 论文中的二项式加一或 epsilon-dropout）。
- `skip_drop`[默认=0.0]
  - 在 boosting 迭代期间跳过 dropout 过程的概率。
    - 如果跳过 dropout，则会以与 相同的方式添加新树`gbtree`。
    - 请注意，非零`skip_drop`的优先级高于`rate_drop`或`one_drop`。
  - 范围：[0.0，1.0]

### 线性助推器的参数 ( `booster=gblinear`)[](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-linear-booster-booster-gblinear)

- `lambda`[默认=0，别名：`reg_lambda`]
  - 权重的 L2 正则化项。增加该值将使模型更加保守。标准化为训练示例的数量。
  - NET补充：
    - 用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
    - lambda_bias :在偏置上的L2正则。`缺省值为0`（在L1上没有偏置项的正则，因为L1时偏置不重要）
- `alpha`[默认=0，别名：`reg_alpha`]
  - 权重的 L1 正则化项。增加该值将使模型更加保守。标准化为训练示例的数量。
  - NET补充：
    - 当数据维度极高时可以使用，使得算法运行更快。
- `updater`[默认= `shotgun`]
  - 选择拟合线性模型的算法
    - `shotgun`：基于霰弹枪算法的并行坐标下降算法。使用“hogwild”并行性，因此每次运行都会产生不确定的解决方案。
    - `coord_descent`：普通坐标下降算法。也是多线程的，但仍然产生确定性的解决方案。
- `feature_selector`[默认= `cyclic`]
  - 特征选择和排序方法
    - `cyclic`：通过一次循环一个功能来进行确定性选择。
    - `shuffle``cyclic`：与每次更新之前类似，但具有随机特征洗牌。
    - `random`：随机（带替换）坐标选择器。
    - `greedy`：选择梯度幅值最大的坐标。它具有`O(num_feature^2)`复杂性。它是完全确定性的。它允许通过设置参数将选择限制为`top_k`每组具有最大单变量权重变化的特征`top_k`。这样做会将复杂性降低到`O(num_feature*top_k)`.
    - `thrifty`：节俭、近似贪婪的特征选择器。在循环更新之前，按单变量权重变化的降序对特征进行重新排序。该操作是多线程的，并且是二次贪婪选择的线性复杂度近似。它允许通过设置参数将选择限制为`top_k`每组具有最大单变量权重变化的特征`top_k`。
- `top_k`[默认=0]
  - `greedy`要在特征选择器中选择的顶级特征的数量`thrifty`。值为 0 表示使用所有功能。

## 学习任务参数

明确学习任务和相应的学习目标。目标选项如下：

- `objective`[默认=reg:平方误差]

  - `reg:squarederror`：平方损失回归。

  - `reg:squaredlogerror`：平方对数损失回归12[log(pred+1)−log(label+1)]2。所有输入标签都必须大于-1。另外，请参阅指标`rmsle`以了解此目标可能存在的问题。

  - `reg:logistic`：逻辑回归。

  - `reg:pseudohubererror`：带有伪 Huber 损失的回归，这是绝对损失的二次可微替代方案。

  - `reg:absoluteerror`：带有 L1 误差的回归。当使用树模型时，叶子值在树构建后刷新。如果用于分布式训练，叶子值将计算为所有worker的平均值，这不能保证是最优的。

    *1.7.0 版本中的新增功能。*

  - `reg:quantileerror`：分位数损失，也称为。有关其参数，请参阅后面的部分；有关工作示例，请参阅[分位数回归。](https://xgboost.readthedocs.io/en/latest/python/examples/quantile_regression.html#sphx-glr-python-examples-quantile-regression-py)`pinball loss`

    *2.0.0 版本中的新增功能。*

  - `binary:logistic`：二元分类的逻辑回归，输出概率

  - `binary:logitraw`：二元分类的逻辑回归，逻辑变换前的输出分数

  - `binary:hinge`：二元分类的铰链损失。这会做出 0 或 1 的预测，而不是产生概率。

  - `count:poisson`：计数数据的泊松回归，泊松分布的输出均值。

    - `max_delta_step`泊松回归中默认设置为 0.7（用于保障优化）

  - `survival:cox`：右删失生存时间数据的 Cox 回归（负值被视为右删失）。请注意，预测是按风险比尺度返回的（即比例风险函数中的 HR = exp(marginal_prediction) ）。`h(t) = h0(t) * HR`

  - `survival:aft`：截尾生存时间数据的加速失效时间模型。有关详细信息，请参阅[加速故障时间的生存分析。](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html)

  - `multi:softmax`：设置XGBoost使用softmax目标进行多类分类，您还需要设置num_class（类数）

  - `multi:softprob`：与softmax相同，但输出一个向量，可以进一步重塑为矩阵。结果包含属于每个类别的每个数据点的预测概率。`ndata * nclass``ndata * nclass`

  - `rank:ndcg`：使用 LambdaMART 执行成对排名，其中[标准化贴现累积收益 (NDCG)](http://en.wikipedia.org/wiki/NDCG)最大化。该目标支持点击数据的位置去偏。

  - `rank:map`：使用 LambdaMART 执行成对排名，其中[平均精度 (MAP)](http://en.wikipedia.org/wiki/Mean_average_precision#Mean_average_precision)最大化

  - `rank:pairwise`：使用 LambdaRank 使用Ranknet目标执行成对排名。

  - `reg:gamma`：带有对数链接的伽玛回归。输出是伽马分布的平均值。它可能很有用，例如，用于对保险索赔严重程度进行建模，或用于任何可能呈[伽玛分布的](https://en.wikipedia.org/wiki/Gamma_distribution#Occurrence_and_applications)结果。

  - `reg:tweedie`：带有对数链接的 Tweedie 回归。它可能很有用，例如，对于保险总损失建模，或者对于任何可能是[Tweedie 分布的](https://en.wikipedia.org/wiki/Tweedie_distribution#Occurrence_and_applications)结果。

- `base_score`

  - 所有实例的初始预测分数，全局偏差
  - 在训练之前，会自动估计所选目标的参数。要禁用估计，请指定实数参数。
  - 对于足够数量的迭代，更改该值不会产生太大影响。

- `eval_metric`[根据目标默认]

  - 验证数据的评估指标，将根据目标分配默认指标（回归的rmse，分类的logloss，平均精度等`rank:map`）

  - 用户可以添加多个评价指标。Python 用户：记住将指标作为参数对列表而不是映射传递，这样后者`eval_metric`就不会覆盖之前的指标

  - 下面列出了选择：

    - `rmse`：[均方根误差](http://en.wikipedia.org/wiki/Root_mean_square_error)

    - `rmsle`：均方根对数误差：1N[log(pred+1)−log(label+1)]2。目标的默认指标`reg:squaredlogerror`。该指标减少了数据集中异常值产生的错误。但由于`log`使用了函数，当预测值小于-1时`rmsle`可能会输出。其他要求`nan`请参见。`reg:squaredlogerror`

    - `mae`：[平均绝对误差](https://en.wikipedia.org/wiki/Mean_absolute_error)

    - `mape`：[平均绝对百分比误差](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error)

    - `mphe`：[表示伪 Huber 错误](https://en.wikipedia.org/wiki/Huber_loss)。目标的默认指标`reg:pseudohubererror`。

    - `logloss`：[负对数似然](http://en.wikipedia.org/wiki/Log-likelihood)

    - `error`：二元分类错误率。其计算方法为。对于预测，评估会将预测值大于0.5的实例视为正实例，将其他实例视为负实例。`#(wrong cases)/#(all cases)`

    - `error@t`：可以通过通过“t”提供数值来指定不同于 0.5 的二元分类阈值。

    - `merror`：多类分类错误率。其计算方法为。`#(wrong cases)/#(all cases)`

    - `mlogloss`：[多类对数损失](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)。

    - `auc`：[接收器工作特性曲线下的区域](https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve)。可用于分类和学习排名任务。

      - 当与二元分类一起使用时，目标应该是`binary:logistic`或类似的处理概率的函数。
      - 当与多类分类一起使用时， Objective 应该`multi:softprob`代替`multi:softmax`，因为后者不输出概率。此外，AUC 是通过 1-vs-rest 以及按类别患病率加权的参考类别来计算的。
      - 当与 LTR 任务一起使用时，AUC 是通过比较文档对来计算正确排序的对的数量。这对应于成对学习排序。该实施存在一些问题，即群体和分布式工作人员的平均 AUC 没有明确定义。
      - 在单台机器上，AUC 计算是准确的。在分布式环境中，AUC 是每个节点上训练行 AUC 的加权平均值 - 因此，分布式 AUC 是对工作人员之间的数据分布敏感的近似值。如果精度和可重复性很重要，请在分布式环境中使用另一个指标。
      - 当输入数据集仅包含负样本或正样本时，输出为NaN。该行为是实现定义的，例如，`scikit-learn`返回0.5反而。

    - `aucpr`：[PR 曲线下的面积](https://en.wikipedia.org/wiki/Precision_and_recall)。可用于分类和学习排名任务。

      XGBoost 1.6之后，在分类问题中使用的要求和限制都`aucpr`与 类似`auc`。对于排名任务，只有二元相关性标签y∈[0,1]是支持的。与 不同，使用连续插值计算精确召回曲线下的*插值面积。*`map (mean average precision)``aucpr`

    - `pre`：精度为k。仅支持学习对任务进行排名。

    - `ndcg`：[标准化贴现累积增益](http://en.wikipedia.org/wiki/NDCG)

    - `map`：[平均精度](http://en.wikipedia.org/wiki/Mean_average_precision#Mean_average_precision)

      平均精度定义为：

      AP@l=1min(l,N)∑k=1lP@k⋅I(k)

      在哪里I(k)是一个指标函数，等于1当文档位于k是相关的并且0否则。这P@k是精度为k， 和N是相关文档的总数。最后，平均精度定义为所有查询的加权平均值。

    - `ndcg@n`, `map@n`, `pre@n`:n可以指定为整数以截断列表中的顶部位置以进行评估。

    - `ndcg-`, `map-`, `ndcg@n-`, `map@n-`：在XGBoost中，NDCG和MAP评估没有任何正样本的列表的分数为1。通过在评估指标名称后添加“-”，我们可以要求 XGBoost 将这些分数评估为0在某些条件下保持一致。

    - `poisson-nloglik`：泊松回归的负对数似然

    - `gamma-nloglik`：伽玛回归的负对数似然

    - `cox-nloglik`：Cox 比例风险回归的负部分对数似然

    - `gamma-deviance`：伽玛回归的残余偏差

    - `tweedie-nloglik`：Tweedie 回归的负对数似然（在指定的`tweedie_variance_power`参数值处）

    - `aft-nloglik`：加速故障时间模型的负对数可能性。有关详细信息，请参阅[加速故障时间的生存分析。](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html)

    - `interval-regression-accuracy`：预测标签属于区间删失标签的数据点的分数。仅适用于区间删失数据。有关详细信息，请参阅[加速故障时间的生存分析。](https://xgboost.readthedocs.io/en/latest/tutorials/aft_survival_analysis.html)

- `seed`[默认=0]

  - 随机数种子。该参数在 R 包中被忽略，请使用set.seed()代替。

- `seed_per_iteration`[默认= `false`]

  - 通过迭代器编号确定种子 PRNG。

### Tweedie 回归的参数 ( `objective=reg:tweedie`)[](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-tweedie-regression-objective-reg-tweedie)

- `tweedie_variance_power`[默认=1.5]
  - 控制 Tweedie 分布方差的参数`var(y) ~ E(y)^tweedie_variance_power`
  - 范围：(1,2)
  - 设置接近 2 以转向伽玛分布
  - 设置接近 1 以转向泊松分布。

### 使用 Pseudo-Huber 的参数 ( `reg:pseudohubererror`)[](https://xgboost.readthedocs.io/en/latest/parameter.html#parameter-for-using-pseudo-huber-reg-pseudohubererror)

- `huber_slope`：用于 Pseudo-Huber 损失的参数，用于定义δ学期。[默认值 = 1.0]

### 使用分位数损失的参数 ( `reg:quantileerror`)[](https://xgboost.readthedocs.io/en/latest/parameter.html#parameter-for-using-quantile-loss-reg-quantileerror)

- `quantile_alpha`：标量或目标分位数列表。

  > *2.0.0 版本中的新增功能。*

### 使用 AFT 生存损失 ( `surviv al:aft`) 和 AFT 度量的负对数似然 ( `aft-nloglik`)的参数[](https://xgboost.readthedocs.io/en/latest/parameter.html#parameter-for-using-aft-survival-loss-survival-aft-and-negative-log-likelihood-of-aft-metric-aft-nloglik)

- `aft_loss_distribution`：概率密度函数、`normal`、`logistic`、 或`extreme`。



### 用于学习排名的参数 ( `rank:ndcg`, `rank:map`, `rank:pairwise`)[](https://xgboost.readthedocs.io/en/latest/parameter.html#parameters-for-learning-to-rank-rank-ndcg-rank-map-rank-pairwise)

这些是学习排序任务特有的参数。请参阅[学习排名](https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html)以获得深入的解释。

- `lambdarank_pair_method`[默认= `mean`]

  如何构建配对以进行配对学习。

  - `mean`：`lambdarank_num_pair_per_sample`查询列表中每个文档的样本对。
  - `topk`：专注于顶级`lambdarank_num_pair_per_sample`文档。构造|query|`lambdarank_num_pair_per_sample`模型排名靠前的每个文档的对。

- `lambdarank_num_pair_per_sample`[范围=[1,∞]]

  当pair方法为 时，它指定为每个文档采样的对的数量`mean`，或者当pair方法为 时，它指定查询的截断级别`topk`。例如，要使用 进行训练`ndcg@6`，请设置`lambdarank_num_pair_per_sample`为6和`lambdarank_pair_method`到`topk`。

- `lambdarank_unbiased`[默认= `false`]

  指定是否需要对输入点击数据进行去偏。

- `lambdarank_bias_norm`[默认值 = 2.0]

  Lp位置去偏标准化，默认为L2。`lambdarank_unbiased`仅当设置为 true时相关。

- `ndcg_exp_gain`[默认= `true`]

  我们是否应该使用指数增益函数`NDCG`. 的增益函数有两种形式`NDCG`，一种是直接使用相关值，另一种是使用2rel−1重点抓好相关文件的检索。当`ndcg_exp_gain`为 true 时（默认），相关度不能大于 31。

## 命令行参数

以下参数仅在 XGBoost 控制台版本中使用

- `num_round`
  - boost的轮数
- `data`
  - 训练数据的路径
- `test:data`
  - 测试数据进行预测的路径
- `save_period`[默认=0]
  - 保存模型的时间段。设置`save_period=10`意味着每 10 轮 XGBoost 就会保存模型。设置为0意味着在训练过程中不保存任何模型。
- `task`[默认= `train`] 选项：`train`, `pred`, `eval`,`dump`
  - `train`：使用数据进行训练
  - `pred`：对测试数据进行预测
  - `eval`：用于评估指定的统计数据`eval[name]=filename`
  - `dump`：用于将学习到的模型转储为文本格式
- `model_in`[默认=空]
  - 、 、`test`任务所需的输入模型路径。如果在训练中指定，XGBoost 将从输入模型继续训练。`eval``dump`
- `model_out`[默认=空]
  - 训练完成后输出模型的路径。如果未指定，XGBoost 将输出名称为`0003.model`where`0003`是 boosting rounds 的文件。
- `model_dir`[默认= `models/`]
  - 训练时保存的模型的输出目录
- `fmap`
  - 特征图，用于转储模型
- `dump_format`[默认= `text`] 选项：`text`,`json`
  - 模型转储文件的格式
- `name_dump`[默认= `dump.txt`]
  - 模型转储文件的名称
- `name_pred`[默认= `pred.txt`]
  - 预测文件名，用于pred模式
- `pred_margin`[默认=0]
  - 预测边际而不是变换概率









# XGBoost-API-NET补充





https://www.cnblogs.com/wj-1314/p/9402324.html



在运行Xgboost之前，必须设置三种类型参数：general parameters，booster parameters和task parameters：

　　**通用参数（General Parameters）**：该参数控制在提升（boosting）过程中使用哪种booster，常用的booster有树模型（tree）和线性模型（linear model）

　　**Booster参数（Booster Parameters）**：这取决于使用哪种booster

　　**学习目标参数（Task Parameters）**：控制学习的场景，例如在回归问题中会使用不同的参数控制排序



## Xgboost的模型参数



Xgboost使用key-value字典的方式存储参数

示例

```
# xgboost模型
params = {
    'booster':'gbtree',
    'objective':'multi:softmax',   # 多分类问题
    'num_class':10,  # 类别数，与multi softmax并用
    'gamma':0.1,    # 用于控制是否后剪枝的参数，越大越保守，一般0.1 0.2的样子
    'max_depth':12,  # 构建树的深度，越大越容易过拟合
    'lambda':2,  # 控制模型复杂度的权重值的L2 正则化项参数，参数越大，模型越不容易过拟合
    'subsample':0.7, # 随机采样训练样本
    'colsample_bytree':3,# 这个参数默认为1，是每个叶子里面h的和至少是多少
    # 对于正负样本不均衡时的0-1分类而言，假设h在0.01附近，min_child_weight为1
    #意味着叶子节点中最少需要包含100个样本。这个参数非常影响结果，
    # 控制叶子节点中二阶导的和的最小值，该参数值越小，越容易过拟合
    'silent':0,  # 设置成1 则没有运行信息输入，最好是设置成0
    'eta':0.007,  # 如同学习率
    'seed':1000,
    'nthread':7,  #CPU线程数
    #'eval_metric':'auc'
}
```





**通用参数，booster参数-tree和booster参数-linear已经补充到官网介绍中**

**学习目标参数未补充过去**





### 通用参数

- - booster [default=gbtree] 
    - 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。`缺省值为gbtree`
  - silent [default=0] 
    - 取0时表示打印出运行时信息，取1时表示以缄默方式运行，不打印运行时的信息。`缺省值为0`
    - 建议取0，过程中的输出数据有助于理解模型以及调参。另外实际上我设置其为1也通常无法缄默运行。。
  - nthread [default to maximum number of threads available if not set] 
    - XGBoost运行时的线程数。`缺省值是当前系统可以获得的最大线程数`
    - 如果你希望以最大速度运行，建议不设置这个参数，模型将自动获得最大线程
  - num_pbuffer [set automatically by xgboost, no need to be set by user] 
    - size of prediction buffer, normally set to number of training instances. The buffers are used to save the prediction results of last boosting step.
  - num_feature [set automatically by xgboost, no need to be set by user] 
    - boosting过程中用到的特征维数，设置为特征个数。`XGBoost会自动设置，不需要手工设置`



### booster参数-tree 

- - eta [default=0.3] 
    - 为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。 eta通过缩减特征的权重使提升计算过程更加保守。`缺省值为0.3`
    - 取值范围为：[0,1]
    - 通常最后设置eta为0.01~0.2
  - gamma [default=0] 
    - minimum loss reduction required to make a further partition on a leaf node of the tree. the larger, the more conservative the algorithm will be.
    - range: [0,∞]
    - 模型在默认情况下，对于一个节点的划分只有在其loss function 得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值
    - gamma值使得算法更conservation，且其值依赖于loss function ，在模型中应该进行调参。
  - max_depth [default=6] 
    - 树的最大深度。`缺省值为6`
    - 取值范围为：[1,∞]
    - 指树的最大深度
    - 树的深度越大，则对数据的拟合程度越高（过拟合程度也越高）。即该参数也是控制过拟合
    - 建议通过交叉验证（xgb.cv ) 进行调参
    - 通常取值：3-10
  - min_child_weight [default=1] 
    - 孩子节点中最小的样本权重和。如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。在现行回归模型中，这个参数是指建立每个模型所需要的最小样本数。该成熟越大算法越conservative。即调大这个参数能够控制过拟合。
    - 取值范围为: [0,∞]
  - max_delta_step [default=0] 
    - Maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update
    - 取值范围为：[0,∞]
    - 如果取值为0，那么意味着无限制。如果取为正数，则其使得xgboost更新过程更加保守。
    - 通常不需要设置这个值，但在使用logistics 回归时，若类别极度不平衡，则调整该参数可能有效果
  - subsample [default=1] 
    - 用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。
    - 取值范围为：(0,1]
  - colsample_bytree [default=1] 
    - 在建立树时对特征随机采样的比例。`缺省值为1`
    - 取值范围：(0,1]
  - colsample_bylevel[default=1]
    - 决定每次节点划分时子样例的比例
    - 通常不使用，因为subsample和colsample_bytree已经可以起到相同的作用了
  - scale_pos_weight[default=0]
    - A value greater than 0 can be used in case of high class imbalance as it helps in faster convergence.
    - 大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛

### booster参数-linear 

- ambda [default=0] 
  - L2 正则的惩罚系数
  - 用于处理XGBoost的正则化部分。通常不使用，但可以用来降低过拟合
- alpha [default=0] 
  - L1 正则的惩罚系数
  - 当数据维度极高时可以使用，使得算法运行更快。
- lambda_bias 
  - 在偏置上的L2正则。`缺省值为0`（在L1上没有偏置项的正则，因为L1时偏置不重要）

### 学习目标参数

　　这个参数是来控制理想的优化目标和每一步结果的度量方法。

- - objective [ default=reg:linear ] 
    - 定义学习任务及相应的学习目标，可选的目标函数如下：
    - “reg:linear” –线性回归。
    - “reg:logistic” –逻辑回归。
    - “binary:logistic” –二分类的逻辑回归问题，输出为概率。
    - “binary:logitraw” –二分类的逻辑回归问题，输出的结果为wTx。
    - “count:poisson” –计数问题的poisson回归，输出结果为poisson分布。
    - 在poisson回归中，max_delta_step的缺省值为0.7。(used to safeguard optimization)
    - “multi:softmax” –让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参数num_class（类别个数）
    - “multi:softprob” –和softmax一样，但是输出的是ndata * nclass的向量，可以将该向量reshape成ndata行nclass列的矩阵。每行数据表示样本所属于每个类别的概率。
    - “rank:pairwise” –set XGBoost to do ranking task by minimizing the pairwise loss
  - base_score [ default=0.5 ] 
    - the initial prediction score of all instances, global bias
  - eval_metric [ default according to objective ] 
    - 校验数据所需要的评价指标，不同的目标函数将会有缺省的评价指标（rmse for regression, and error for classification, mean average precision for ranking）
    - 用户可以添加多种评价指标，对于Python用户要以list传递参数对给程序，而不是map参数list参数不会覆盖’eval_metric’
    - The choices are listed below:
    - “rmse”: [root mean square error](http://en.wikipedia.org/wiki/Root_mean_square_error)
    - “logloss”: negative [log-likelihood](http://en.wikipedia.org/wiki/Log-likelihood)
    - “error”: Binary classification error rate. It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
    - “merror”: Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
    - “mlogloss”: Multiclass logloss
    - “[auc](https://www.baidu.com/s?wd=auc&tn=24004469_oem_dg&rsv_dl=gh_pl_sl_csd)”: [Area under the curve](http://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_curve) for ranking evaluation.
    - “ndcg”:[Normalized Discounted Cumulative Gain](http://en.wikipedia.org/wiki/NDCG)
    - “map”:[Mean average precision](http://en.wikipedia.org/wiki/Mean_average_precision#Mean_average_precision)
    - “ndcg@n”,”map@n”: n can be assigned as an integer to cut off the top positions in the lists for evaluation.
    - “ndcg-“,”map-“,”ndcg@n-“,”map@n-“: In XGBoost, NDCG and MAP will evaluate the score of a list without any positive samples as 1. By adding “-” in the evaluation metric XGBoost will evaluate these score as 0 to be consistent under some conditions. 
      training repeatively
  - seed [ default=0 ] 
    - 随机数的种子。`缺省值为0`
    - 可以用于产生可重复的结果（每次取一样的seed即可得到相同的随机划分）

![img](https://img2018.cnblogs.com/blog/1226410/201901/1226410-20190129171411164-1583579393.png)



## Xgboost.train方法

```
xgboost.train(params,dtrain,num_boost_round=10,evals(),obj=None,
feval=None,maximize=False,early_stopping_rounds=None,evals_result=None,
verbose_eval=True,learning_rates=None,xgb_model=None)
```

　　

　　parms：这是一个字典，里面包含着训练中的参数关键字和对应的值，形式是parms = {'booster':'gbtree','eta':0.1}

　　dtrain：训练的数据

　　num_boost_round：这是指提升迭代的个数

　　evals：这是一个列表，用于对训练过程中进行评估列表中的元素。形式是evals = [(dtrain,'train'),(dval,'val')] 或者是 evals =[(dtrain,'train')] ，对于第一种情况，它使得我们可以在训练过程中观察验证集的效果。

　　obj ：自定义目的函数

　　feval：自定义评估函数

　　maximize：是否对评估函数进行最大化

　　early_stopping_rounds：早起停止次数，假设为100，验证集的误差迭代到一定程度在100次内不能再继续降低，就停止迭代。这要求evals里至少有一个元素，如果有多个，按照最后一个去执行。返回的是最后的迭代次数（不是最好的）。如果early_stopping_rounds存在，则模型会生成三个属性，bst.best_score ,bst.best_iteration和bst.best_ntree_limit

　　evals_result：字典，存储在watchlist中的元素的评估结果

　　verbose_eval（可以输入布尔型或者数值型）：也要求evals里至少有一个元素，如果为True，则对evals中元素的评估结果会输出在结果中；如果输入数字，假设为5，则每隔5个迭代输出一次。

　　learning_rates：每一次提升的学习率的列表

　　xgb_model：在训练之前用于加载的xgb_model





### 绘图API

https://www.biaodianfu.com/xgboost.html



**xgboost.plot_importance()：绘制特征重要性**

```
xgboost.plot_importance(booster, ax=None, height=0.2, xlim=None, ylim=None,
       title='Feature importance', xlabel='F score', ylabel='Features',
       importance_type='weight', max_num_features=None, grid=True, 
       show_values=True, **kwargs)   
      
```



参数：

- booster： 一个Booster对象， 一个 XGBModel 对象，或者由get_fscore() 返回的字典
- ax： 一个matplotlib Axes 对象。特征重要性将绘制在它上面。如果为None，则新建一个Axes
- grid： 一个布尔值。如果为True，则开启axes grid
- importance_type： 一个字符串，指定了特征重要性的类别。参考get_fscore()
- max_num_features： 一个整数，指定展示的特征的最大数量。如果为None，则展示所有的特征
- height： 一个浮点数，指定bar 的高度。它传递给barh()
- xlim： 一个元组，传递给xlim()
- ylim： 一个元组，传递给ylim()
- title： 一个字符串，设置Axes 的标题。默认为”Feature importance”。 如果为None，则没有标题
- xlabel： 一个字符串，设置Axes 的X 轴标题。默认为”F score”。 如果为None，则X 轴没有标题
- ylabel：一个字符串，设置Axes 的Y 轴标题。默认为”Features”。 如果为None，则Y 轴没有标题
- show_values： 一个布尔值。如果为True，则在绘图上展示具体的值。
- kwargs： 关键字参数，用于传递给barh()



**xgboost.plot_tree()： 绘制指定的子树**



```
xgboost.plot_tree(booster, fmap='', num_trees=0, rankdir='UT', ax=None, **kwargs)
```



参数：

- booster： 一个Booster对象， 一个 XGBModel 对象
- fmap： 一个字符串，给出了feature map 文件的文件名
- num_trees： 一个整数，制定了要绘制的子数的编号。默认为 0
- rankdir： 一个字符串，它传递给graphviz的graph_attr
- ax： 一个matplotlib Axes 对象。特征重要性将绘制在它上面。如果为None，则新建一个Axes
- kwargs： 关键字参数，用于传递给graphviz 的graph_attr





**xgboost.to_graphviz()： 转换指定的子树成一个graphviz 实例**

在IPython中，可以自动绘制graphviz 实例；否则你需要手动调用graphviz 对象的.render() 方法来绘制。

```
xgboost.to_graphviz(booster, fmap='', num_trees=0, rankdir='UT', yes_color='#0000FF',
     no_color='#FF0000', **kwargs)
     
```

参数：

- yes_color： 一个字符串，给出了满足node condition 的边的颜色
- no_color： 一个字符串，给出了不满足node condition 的边的颜色
- 其它参数参考 plot_tree()



示例：

```
import xgboost as xgt
import pandas as pd
from sklearn.model_selection import train_test_split
from matplotlib.pylab import plot as plt

_label_map = {
    # 'Iris-setosa':0, #经过裁剪的，去掉了 iris 中的 setosa 类
    'Iris-versicolor': 0,
    'Iris-virginica': 1
}


class PlotTest:
    def __init__(self):
        df = pd.read_csv('./data/iris.csv')
        _feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        x = df[_feature_names]
        y = df['Class'].map(lambda x: _label_map[x])

        train_X, test_X, train_Y, test_Y = train_test_split(x, y,
                                                            test_size=0.3, stratify=y, shuffle=True, random_state=1)
        self._train_matrix = xgt.DMatrix(data=train_X, label=train_Y,
                                         feature_names=_feature_names,
                                         feature_types=['float', 'float', 'float', 'float'])
        self._validate_matrix = xgt.DMatrix(data=test_X, label=test_Y,
                                            feature_names=_feature_names,
                                            feature_types=['float', 'float', 'float', 'float'])

    def plot_test(self):
        params = {
            'booster': 'gbtree',
            'eta': 0.01,
            'max_depth': 5,
            'tree_method': 'exact',
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'error', 'auc']
        }
        eval_rst = {}
        booster = xgt.train(params, self._train_matrix,
                            num_boost_round=20, evals=([(self._train_matrix, 'valid1'),
                                                        (self._validate_matrix, 'valid2')]),
                            early_stopping_rounds=5, evals_result=eval_rst, verbose_eval=True)
        xgt.plot_importance(booster)
        plt.show()
```

