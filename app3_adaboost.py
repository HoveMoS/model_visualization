from definitions import *
from sklearn.ensemble import AdaBoostClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)
st.sidebar.subheader("请选择模型参数:sunglasses:")


n_estimators = st.sidebar.slider(label = 'n_estimators', min_value = 1,
                          max_value = 100 ,
                          value = 50,
                          step = 1)


learning_rate = st.sidebar.slider(label = 'learning_rate',  min_value = 0.1,
                          max_value = 1.1,
                          value = 1.0,
                          step = 0.1)







                                 
st.header('Adaboost-parameter-tuning-with-streamlit')


# 加载数据
breast_cancer = load_breast_cancer()
data = breast_cancer.data
target = breast_cancer.target

# 划分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2,random_state=10)




# 模型训练

model = AdaBoostClassifier(base_estimator=None, n_estimators=n_estimators,learning_rate=learning_rate,random_state=1)
model.fit(X_train, y_train)


probs = model.predict(X_test)  # 输出的是概率结果




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