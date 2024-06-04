import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

@st.cache_data
def load_data():
    data = pd.read_csv('BankChurners.csv')
    data = data.drop(columns=['CLIENTNUM'])
    # Преобразуем целевую переменную в числовой формат
    data['Attrition_Flag'] = data['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
    return data

@st.cache_resource
def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    data_out = data_in.copy()
    # Получим список номинальных признаков
    nominal_features = data_out.select_dtypes(include=['object']).columns
    data_out = pd.get_dummies(data_out, columns=nominal_features, drop_first=True)

    # Масштабируем числовые признаки
    scale_cols = data_out.select_dtypes(include=[np.number]).columns
    sc = MinMaxScaler()
    data_out[scale_cols] = sc.fit_transform(data_out[scale_cols])
    
    X = data_out.drop(columns=['Attrition_Flag'])
    y = data_out['Attrition_Flag']
    return X, y

# Загрузка и предварительная обработка данных
data = load_data()
data_X, data_y = preprocess_data(data)

# Интерфейс пользователя
st.sidebar.header('Метод ближайших соседей')
cv_slider = st.sidebar.slider('Количество фолдов:', min_value=3, max_value=10, value=3, step=1)
step_slider = st.sidebar.slider('Шаг для соседей:', min_value=1, max_value=50, value=10, step=1)

# Подбор гиперпараметра
n_range_list = list(range(1, 100, step_slider)) # Установите максимальное значение в соответствии с вашими данными
n_range = np.array(n_range_list)
tuned_parameters = [{'n_neighbors': n_range}]

clf_gs = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=cv_slider, scoring='accuracy')
clf_gs.fit(data_X, data_y)

best_params = clf_gs.best_params_
# Получение значения accuracy для лучшего гиперпараметра
best_accuracy = clf_gs.cv_results_['mean_test_score'][clf_gs.best_index_]

# Получение предсказанных значений для лучшей модели
best_model = clf_gs.best_estimator_
predictions = best_model.predict(data_X)

# Вычисление precision и recall
precision = precision_score(data_y, predictions)
recall = recall_score(data_y, predictions)

# Вывод оценок на интерфейс
st.subheader('Оценка модели')
st.write('Лучшее значение гиперпараметра (Количество соседей):', best_params['n_neighbors'])
st.write('Accuracy:', best_accuracy)
st.write('Precision:', precision)
st.write('Recall:', recall)

# Изменение качества на тестовой выборке в зависимости от К-соседей
fig1 = plt.figure(figsize=(7,5))
plt.plot(n_range, clf_gs.cv_results_['mean_test_score'])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Neighbors')
st.pyplot(fig1)
