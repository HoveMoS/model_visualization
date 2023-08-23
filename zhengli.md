

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





# 树助推器的参数





- `gamma`[默认=0，别名：`min_split_loss`]

  - 在树的叶节点上进行进一步划分所需的最小损失减少。越大`gamma`，算法越保守。
  - 范围：[0,无穷大]
  - NET补充：
    - 模型在默认情况下，对于一个节点的划分只有在其loss function 得到结果大于0的情况下才进行，而gamma 给定了所需的最低loss function的值
    - gamma值使得算法更conservation，且其值依赖于loss function ，在模型中应该进行调参。



## 整数滑动条









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

  

  

- `lambda`[默认=1，别名：`reg_lambda`]

  - 权重的 L2 正则化项。增加该值将使模型更加保守。
  - 范围：[0,∞]

- `alpha`[默认=0，别名：`reg_alpha`]

  - 权重的 L1 正则化项。增加该值将使模型更加保守。
  - 范围：[0,∞]

  





- `max_leaves`[默认=0]

  - 要添加的最大节点数。不被树方法使用`exact`。

- `max_bin`，[默认=256]

  - 仅当`tree_method`设置为`hist`、`approx`或 时才使用`gpu_hist`。
  - 用于存储连续特征的离散箱的最大数量。
  - 增加这个数字可以提高分割的最优性，但代价是增加计算时间。

  

- `num_parallel_tree`，[默认=1]

  - 每次迭代期间构建的并行树的数量。此选项用于支持增强随机森林。





## 字符选项





`sampling_method`[默认= `uniform`]

- 用于对训练实例进行采样的方法。
- `uniform`：每个训练实例被选择的概率相等。通常设置 `subsample`>= 0.5 以获得良好结果。
- `gradient_based`：每个训练实例的选择概率与 梯度的*正则化绝对值*成正比（更具体地说，g2+λh2）。 `subsample`可以设置为低至 0.1，而不会损失模型精度。`tree_method`请注意，仅当设置为 时才支持此采样方法`gpu_hist`；其他树方法仅支持`uniform`采样。







`tree_method`字符串 [默认= `auto`]

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







- `updater`

  - 逗号分隔的字符串，定义要运行的树更新程序的序列，提供构建和修改树的模块化方法。这是一个高级参数，通常根据一些其他参数自动设置。然而，它也可以由用户显式设置。存在以下更新程序：
    - `grow_colmaker`：非分布式基于列的树构建。
    - `grow_histmaker`：基于直方图计数的全局提议的基于行的数据分割的分布式树构建。
    - `grow_quantile_histmaker`：使用量化直方图生长树。
    - `grow_gpu_hist`：使用 GPU 种植树。
    - `sync`：同步所有分布式节点中的树。
    - `refresh`：根据当前数据刷新树的统计数据和/或叶子值。请注意，不执行数据行的随机子采样。
    - `prune`：修剪 loss < min_split_loss （或 gamma）的分割以及深度大于 的节点`max_depth`。

  





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

  







`multi_strategy`, [默认 = `one_output_per_tree`]

*2.0.0 版本中的新增功能。*

笔记

该参数正在处理中。

- 该策略用于训练多目标模型，包括多目标回归和多类分类。有关详细信息，请参阅[多个输出](https://xgboost.readthedocs.io/en/latest/tutorials/multioutput.html)。
  - `one_output_per_tree`：每个目标一个模型。
  - `multi_output_tree`：使用多目标树。





### GPU

#### 

`predictor`, [默认= `auto`]

- 要使用的预测器算法的类型。提供相同的结果，但允许使用 GPU 或 CPU。
  - `auto`：根据启发式配置预测器。
  - `cpu_predictor`：多核CPU预测算法。
  - `gpu_predictor`：使用 GPU 进行预测。`tree_method`当是时使用`gpu_hist`。当`predictor`设置为默认值时`auto`，`gpu_hist`树方法能够提供基于 GPU 的预测，而无需将训练数据复制到 GPU 内存。如果`gpu_predictor`明确指定，则所有数据都会复制到 GPU 中，仅建议用于执行预测任务。

## 浮点数



`eta`[默认=0.3，别名：`learning_rate`]

- 更新中使用步长收缩来防止过度拟合。在每个boosting步骤之后，我们可以直接得到新特征的权重，并`eta`缩小特征权重以使boosting过程更加保守。
- 范围：[0,1]
- NET补充：
  - 为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算之后，算法会直接获得新特征的权重。 eta通过缩减特征的权重使提升计算过程更加保守。`缺省值为0.3`
  - 通常最后设置eta为0.01~0.2







`subsample`[默认=1]

- 训练实例的子样本比率。将其设置为 0.5 意味着 XGBoost 将在生长树之前随机采样一半的训练数据。这将防止过度拟合。子采样将在每次提升迭代中发生一次。
- 范围：(0,1]
- NET补充：用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中抽取出50%的子样本建立树模型，这能够防止过拟合。









## 其他





- `scale_pos_weight`[默认=1]

  - 控制正负权重的平衡，对于不平衡的类很有用。要考虑的典型值：。有关更多讨论，请参阅[参数调整。](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html)另请参阅 Higgs Kaggle 竞赛演示，了解示例：[R](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-train.R)、[py1](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-numpy.py)、[py2](https://github.com/dmlc/xgboost/blob/master/demo/kaggle-higgs/higgs-cv.py)、[py3](https://github.com/dmlc/xgboost/blob/master/demo/guide-python/cross_validation.py)。`sum(negative instances) / sum(positive instances)`
  - NET补充：大于0的取值可以处理类别不平衡的情况。帮助模型更快收敛
- `refresh_leaf`[默认=1]
  - 这是更新器的一个参数`refresh`。当此标志为 1 时，树叶和树节点的统计信息都会更新。当它为 0 时，仅更新节点统计信息。





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









- `monotone_constraints`
  - 变量单调性的约束。有关详细信息，请参阅[单调约束。](https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html)
- `interaction_constraints`

  - 代表允许交互的交互约束。约束必须以嵌套列表的形式指定，例如，其中每个内部列表是一组允许彼此交互的特征索引。有关详细信息，请参阅[功能交互约束。](https://xgboost.readthedocs.io/en/latest/tutorials/feature_interaction_constraint.html)`[[0, 1], [2, 3, 4]]`





