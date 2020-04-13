# Vamsi Desu 
# GT username: vdesu7 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import validation_curve, GridSearchCV
from sklearn.svm import SVC 

credit_fraud = pd.read_csv("default_credit_card_clients.csv")
# credit_fraud = credit_fraud.sample(10000)
train, test = train_test_split(credit_fraud, test_size=0.2)
train_X, train_Y = train.iloc[:, :-1], train.iloc[:, [-1]]
test_X, test_Y = test.iloc[:, :-1], test.iloc[:, [-1]]
print(train_X.shape)
print(train_Y.shape)
# clf = RandomForestClassifier(n_estimators=300)
# clf = clf.fit(train_X, train_Y)
# y_pred = clf.predict(test_X)
# print(y_pred.shape)
# print(test_Y.shape)
# print("Accuracy:",metrics.accuracy_score(test_Y, y_pred))

# param_range = [50,100,150,200,250,300,350,400]
# train_scoreNum, test_scoreNum = validation_curve(
#                                 RandomForestClassifier(),
#                                 X = train_X, y = train_Y.values.ravel(), 
#                                 param_name = 'n_estimators', 
#                                 param_range = [50,100,150,200,250,300,350,400] , cv = 3)
# train_scores_mean = np.mean(train_scoreNum, axis=1)
# test_scores_mean = np.mean(test_scoreNum, axis=1)

# print(train_scores_mean)
# print(test_scores_mean)

# plt.title("Validation Curve with Random Forests with Bagging Technique")
# plt.xlabel("Number of Estimators/Decision Trees")
# plt.ylabel("Accuracy Score")
# plt.plot(param_range, train_scores_mean, label='Training Score', color='red',markerfacecolor='red')
# plt.plot(param_range, test_scores_mean, label='Testing Data',color='blue',markerfacecolor='blue')
# plt.legend(loc="best")
# plt.show()


# max_depth_param_range = [5,10,15,20,25,30,35,40]
# train_scoreNum, test_scoreNum = validation_curve(
#                                 RandomForestClassifier(),
#                                 X = train_X, y = train_Y.values.ravel(), 
#                                 param_name = 'max_depth', 
#                                 param_range = max_depth_param_range , cv = 3)
# train_scores_mean = np.mean(train_scoreNum, axis=1)
# test_scores_mean = np.mean(test_scoreNum, axis=1)

# print(train_scores_mean)
# print(test_scores_mean)

# plt.title("Validation Curve with Random Forests with Bagging Technique")
# plt.xlabel("Max Depth")
# plt.ylabel("Accuracy Score")
# plt.plot(max_depth_param_range, train_scores_mean, label='Training Score', color='red',markerfacecolor='red')
# plt.plot(max_depth_param_range, test_scores_mean, label='Testing Data',color='blue',markerfacecolor='blue')
# plt.legend(loc="best")
# plt.show()

# min_sample_param_range = [5,10,15,20,25,30]
# train_scoreNum, test_scoreNum = validation_curve(
#                                 RandomForestClassifier(),
#                                 X = train_X, y = train_Y.values.ravel(), 
#                                 param_name = 'min_samples_split', 
#                                 param_range = min_sample_param_range , cv = 3)
# train_scores_mean = np.mean(train_scoreNum, axis=1)
# test_scores_mean = np.mean(test_scoreNum, axis=1)

# print(train_scores_mean)
# print(test_scores_mean)

# plt.title("Validation Curve with Random Forests with Bagging Technique")
# plt.xlabel("Min Sample Split")
# plt.ylabel("Accuracy Score")
# plt.plot(min_sample_param_range, train_scores_mean, label='Training Score', color='red',markerfacecolor='red')
# plt.plot(min_sample_param_range, test_scores_mean, label='Testing Data',color='blue',markerfacecolor='blue')
# plt.legend(loc="best")
# plt.show()

# SVM - Grid Search for Best Hyperparameters 

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(train_X,train_Y.values.ravel())
print(grid.best_estimator_)
print(grid.cv_results_)