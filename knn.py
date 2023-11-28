# Load the data
# url = 'data/biopsy.csv'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean 

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from matplotlib.colors import ListedColormap

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

import seaborn as sns



sc = StandardScaler()
file = r"C:\Users\theos\OneDrive - Novotek AB\Desktop\Studier\Statistical Machine Learning\training_data.csv"
data = pd.read_csv(file, dtype={'ID': str}).dropna().reset_index(drop=True)
print(data)

data['increase_stock'] = np.where(data['increase_stock'] == 'low_bike_demand', 0, 1)

data.drop(columns=['increase_stock'])
data.head()
X = data[['hour_of_day','day_of_week','month','holiday','weekday',
          'summertime','temp','dew','humidity','precip','snow',
          'snowdepth','windspeed','cloudcover','visibility']]

#

y = data['increase_stock']
np.squeeze(y).shape
xnum = np.asarray(X)
ynum = np.asarray(y)

# scores_t = []
# scores_v = []
# k_list= []
# X_train, X_test, y_train, y_test = skl_ms.train_test_split(xnum, ynum,test_size=0.7, random_state=np.random.randint(1,50))
# knn = skl_nb.KNeighborsClassifier(n_neighbors = 17)
# knn.fit(X_train, y_train)


# test_preds = knn.predict(X_test)
# print(test_preds)

# plt.show()                          

# scores_v.append(1 - knn.score(X_train, y_train))
# scores_t.append(1 - knn.score(X_test, y_test))


# for k in range (1,80):
#     X_train, X_test, y_train, y_test = skl_ms.train_test_split(xnum, ynum,test_size=0.7, random_state=np.random.randint(1,50))
#     knn = skl_nb.KNeighborsClassifier(n_neighbors = k)
#     knn.fit(X_train, y_train)
#     k_list.append(k)
#     scores.append(1 - knn.score(X_test, y_test))

# cv = KFold(n_splits=8, random_state=62, shuffle=True)
# X_train, X_test, y_train, y_test = skl_ms.train_test_split(xnum, ynum,test_size=0.92, random_state=60)
# knn = skl_nb.KNeighborsClassifier(n_neighbors = 17)
# knn.fit(X_train, y_train)
# holdout_score = (knn.score(X_train, y_train))

#scores = cross_val_score(knn, X_test, y_test, scoring='accuracy', cv=cv, n_jobs=-1)


# plt.plot(X_train, y_train,marker='o', color = 'red')
# plt.plot(X_test, y_test,marker='o', color = 'blue')
#print(mean(scores))
# plt.title('8 fold Cross Validation ')
# plt.xlabel('Cross Validation')
# plt.ylabel('Accuracy')
# i = np.argmax(scores)
# mean_s = mean(scores)
# y_min = scores[i]
# print(y_min)
# plt.text(i,y_min, y_min)
# plt.plot(i, y_min, marker='o', )
# plt.text(1, holdout_score, 'holdout score')
# plt.plot(1, holdout_score, marker='o', )
# plt.legend(['mean = ' + str(mean_s)], loc = 'upper right')
plt.show()   

# scores = []
# scores_m = []
# k_list = []
# for k in range(1, 60):
#     scores = []
#     test_s = 0.05
#     for _ in range(10):
#         X_train, X_test, y_train, y_test = skl_ms.train_test_split(xnum, ynum,test_size=test_s, random_state=np.random.randint(1,50))
#         knn = skl_nb.KNeighborsClassifier(n_neighbors = k)
#         knn.fit(X_train, y_train)
#         scores.append(1 - knn.score(X_test, y_test))
#         test_s += 0.05
#     k_list.append(k)   
#     scores_m.append(mean(scores))
#     k += 2
# plt.plot(k_list, scores_m)



