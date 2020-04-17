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
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

credit_fraud = pd.read_csv("default_credit_card_clients.csv")
X, Y = credit_fraud.iloc[:, :-1], credit_fraud.iloc[:, [-1]]
# credit_fraud = credit_fraud.sample(10000)
train, test = train_test_split(credit_fraud, test_size=0.2)
train_X, train_Y = train.iloc[:, :-1], train.iloc[:, [-1]]
test_X, test_Y = test.iloc[:, :-1], test.iloc[:, [-1]]
print(train_X.shape)
print(train_Y.shape)
# scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
# train_X = scaler.fit_transform(train_X)
# clf = RandomForestClassifier(n_estimators=300)
# clf = clf.fit(train_X, train_Y)
# y_pred = clf.predict(test_X)
# print(y_pred.shape)
# print(test_Y.shape)
# print("Accuracy:",metrics.accuracy_score(test_Y, y_pred))

# Random Forest - Grid Search for Best Hyperparameters  Time: 20.3min


# n_estimators = [50, 100, 150, 200]
# max_depth = [5, 10, 15, 25]
# min_samples_split = [5, 10,100]
# min_samples_leaf = [1, 2, 5] 

# hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
#               min_samples_split = min_samples_split, 
#              min_samples_leaf = min_samples_leaf)
# gridF = GridSearchCV(RandomForestClassifier(), hyperF, cv = 3, verbose = 2)
# bestF = gridF.fit(train_X, train_Y.values.ravel())
# print(bestF.best_estimator_)
# pd.set_option('display.max_rows', None)
# print(pd.concat([pd.DataFrame(bestF.cv_results_["params"]),pd.DataFrame(bestF.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))

# Add Linear to show linear seprability 
# SVM - Grid Search for Best Hyperparameters  Time: 32 mins 

# param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid', ]}
# param_grid_linear = {'C': [0.1,1, 10], 'gamma': [1,0.1,0.01,0.001],'kernel': [ 'linear' ]}
# grid = GridSearchCV(SVC(),param_grid_linear,refit=True,verbose=2)
# grid.fit(train_X,train_Y.values.ravel())
# print(grid.best_estimator_)
# print(pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))




















# Random Forest - Validation Curve Approach 
# Grid Search is Better Approach for Tuning

# param_range = [1,25,50,75,100,125,150,175,200]
# train_scoreNum, test_scoreNum = validation_curve(
#                                 RandomForestClassifier(),
#                                 X = train_X, y = train_Y.values.ravel(), 
#                                 param_name = 'n_estimators', 
#                                 param_range = param_range , cv = 3)
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
#                                 X = X, y = Y.values.ravel(), 
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
#                                 X = X, y = Y, 
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

# min_sample_param_range = [1, 2, 5] 
# train_scoreNum, test_scoreNum = validation_curve(
#                                 RandomForestClassifier(),
#                                 X = train_X, y = train_Y.values.ravel(), 
#                                 param_name = 'min_samples_leaf', 
#                                 param_range = min_sample_param_range , cv = 3)
# train_scores_mean = np.mean(train_scoreNum, axis=1)
# test_scores_mean = np.mean(test_scoreNum, axis=1)

# print(train_scores_mean)
# print(test_scores_mean)

# plt.title("Validation Curve with Random Forests with Bagging Technique")
# plt.xlabel("Min Sample Leaf")
# plt.ylabel("Accuracy Score")
# plt.plot(min_sample_param_range, train_scores_mean, label='Training Score', color='red',markerfacecolor='red')
# plt.plot(min_sample_param_range, test_scores_mean, label='Testing Data',color='blue',markerfacecolor='blue')
# plt.legend(loc="best")
# plt.show()

gamma = [1,0.1,0.01,0.001]
train_scoreNum, test_scoreNum = validation_curve(
                                SVC(),
                                X = X, y = Y.values.ravel(), 
                                param_name = 'gamma', 
                                param_range = gamma , cv = 3)
train_scores_mean = np.mean(train_scoreNum, axis=1)
test_scores_mean = np.mean(test_scoreNum, axis=1)

print(train_scores_mean)
print(test_scores_mean)

plt.title("SVM")
plt.xlabel("Gamma")
plt.ylabel("Accuracy Score")
plt.plot(gamma, train_scores_mean, label='Training Score', color='red',markerfacecolor='red')
plt.plot(gamma, test_scores_mean, label='Testing Data',color='blue',markerfacecolor='blue')
plt.legend(loc="best")
plt.show()

# def plotValidationCurve(parameter_name, parameter_list):
#     train_scoreNum, test_scoreNum = validation_curve(
#                                 SVC(),
#                                 X = X, y = Y.values.ravel(), 
#                                 param_name = 'gamma', 
#                                 param_range = gamma , cv = 3)
#     train_scores_mean = np.mean(train_scoreNum, axis=1)
#     test_scores_mean = np.mean(test_scoreNum, axis=1)

#     print(train_scores_mean)
#     print(test_scores_mean)

#     plt.title("SVM")
#     plt.xlabel("Gamma")
#     plt.ylabel("Accuracy Score")
#     plt.plot(gamma, train_scores_mean, label='Training Score', color='red',markerfacecolor='red')
#     plt.plot(gamma, test_scores_mean, label='Testing Data',color='blue',markerfacecolor='blue')
#     plt.legend(loc="best")
#     plt.show()
