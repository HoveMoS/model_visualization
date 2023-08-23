from definitions import *

st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.subheader("请选择模型参数:sunglasses:")



eta = st.sidebar.slider(label = 'eta',  min_value = 0.1,
                          max_value = 0.9,
                          value = 0.3,
                          step = 0.1)



num_class = st.sidebar.slider(label = 'num_class', min_value = 2,
                          max_value = 20 ,
                          value = 2,
                          step = 1)

# gamma = st.sidebar.slider(label = 'gamma',  min_value = 0,
#                           max_value = 10,
#                           value = 1,
#                           step = 1)
                          
max_depth = st.sidebar.slider(label = 'max_depth',  min_value = 1,
                          max_value = 55,
                          value = 12,
                          step = 1)


min_child_weight = st.sidebar.slider(label = 'min_child_weight',  min_value = 0,
                          max_value = 10,
                          value = 1,
                          step = 1)


max_delta_step = st.sidebar.slider(label = 'max_delta_step',  min_value = 0,
                          max_value = 10,
                          value = 0,
                          step = 1)


lambda_x = st.sidebar.slider(label = 'lambda',  min_value = 0,
                          max_value = 10,
                          value = 1,
                          step = 1)


alpha = st.sidebar.slider(label = 'alpha',  min_value = 0,
                          max_value = 10,
                          value = 0,
                          step = 1)


max_leaves = st.sidebar.slider(label = 'max_leaves',  min_value = 0,
                          max_value = 10,
                          value = 0,
                          step = 1)


max_bin = st.sidebar.slider(label = 'max_bin',  min_value = 200,
                          max_value = 300,
                          value = 256,
                          step = 1)


num_parallel_tree = st.sidebar.slider(label = 'num_parallel_tree',  min_value = 0,
                          max_value = 10,
                          value = 1,
                          step = 1)



# min_child_weight

# max_delta_step
# lambda_x
# alpha
# max_leaves
# max_bin 
# num_parallel_tree



nthread = st.sidebar.slider(label = 'nthread', min_value = 1,
                          max_value = 8 ,
                          value = 2,
                          step = 1)


# 在侧处出现选择框
sampling_method = st.sidebar.selectbox(
    'sampling_method',
    ('uniform', 'gradient_based'))



# 在右侧空白处出现选择框
# sampling_method = st.selectbox(
#     'sampling_method',
#     ('uniform', 'gradient_based'))



                                 
st.header('Xgboost-parameter-tuning-with-streamlit')


# 加载数据
breast_cancer = load_breast_cancer()
data = breast_cancer.data
target = breast_cancer.target

# 划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2,random_state=10)

# 转换为Dataset数据格式
lgb_train = xgb.DMatrix(X_train, y_train)
lgb_eval = xgb.DMatrix(X_test, y_test)

# 模型训练



# num_class
# eta
# gamma
# max_depth
# nthread

# num_class=2
# gamma=0.1
# max_depth=12
# nthread=4

gamma=0.1


# min_child_weight

# max_delta_step
# lambda_x
# alpha
# max_leaves
# max_bin 
# num_parallel_tree



params = {'num_class': num_class,
            'eta': eta,
            'max_depth': max_depth,
            'min_child_weight': min_child_weight,
            'max_delta_step': max_delta_step,
            'lambda': lambda_x,
            'alpha': alpha,
            'max_leaves': max_leaves,
            'max_bin': max_bin,
            'num_parallel_tree': num_parallel_tree,
            'nthread': nthread, 
            'sampling_method': sampling_method,
            }
#plst = list(params.items())


gbm = xgb.train(params, lgb_train)
lgb_eval = xgb.DMatrix(X_test, y_test)  
probs = gbm.predict(lgb_eval)  # 输出的是概率结果  







fpr, tpr, thresholds = roc_curve(y_test, probs)


st.write('------------------------------------')
st.write('Confusion Matrix:')
st.write(confusion_matrix(y_test, np.where(probs > 0.5, 1, 0)))

st.write('------------------------------------')
st.write('Classification Report:')
report = classification_report(y_test, np.where(probs > 0.5, 1, 0), output_dict=True)
report_matrix = pd.DataFrame(report).transpose()
st.dataframe(report_matrix)

st.write('------------------------------------')
st.write('ROC:')

plot_roc(fpr, tpr)