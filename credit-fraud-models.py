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
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import time

# Random Forest - Training and Tuning 

# RF Default Parameters 
def rf_model_default(train_X, train_Y, test_X, test_Y):
    clf = RandomForestClassifier()
    clf = clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)
    print("Accuracy:",metrics.accuracy_score(test_Y, y_pred))

# RF - Model w/ Best Hyperparameter Values
def rf_model(n_estimators, max_depth, min_samples_split, min_samples_leaf, train_X, train_Y,test_X,test_Y):
    # start_time = time.time()
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf)
    clf = clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)
    print(confusion_matrix(test_Y, y_pred))
    # print(classification_report(test_Y, y_pred))
    print("Accuracy:",metrics.accuracy_score(test_Y, y_pred))
    # print("Time: ", time.time() - start_time) 

# RF - Tuning for Chosen Hyperparameters 
def rf_tuning(n_estimators_list, max_depth_list, samples_split_list, leaf_list,train_X, train_Y):
    hyperF = dict(n_estimators = n_estimators_list, max_depth = max_depth_list,  
                  min_samples_split = samples_split_list, 
                 min_samples_leaf = leaf_list)
    gridF = GridSearchCV(RandomForestClassifier(), hyperF, cv = 3, verbose = 2)
    bestF = gridF.fit(train_X, train_Y.values.ravel())
    print(bestF.best_estimator_)
    pd.set_option('display.max_rows', None)
    print(pd.concat([pd.DataFrame(bestF.cv_results_["params"]),pd.DataFrame(bestF.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))

# SVM - Training and Tuning 

# SVM Default Parameters 
def svm_model_default(train_X, train_Y, test_X, test_Y):
    
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)
    clf = RandomForestClassifier()
    clf = clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)
    print("Accuracy:",metrics.accuracy_score(test_Y, y_pred))
    print("Time: ", time.time() - start_time) 

# SVM - Model w/ Best Hyperparameter Values
def svm_model(c, gamma, kernel, train_X, train_Y, test_X, test_Y):
    start_time = time.time()
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)
    clf = SVC(C = c, gamma = gamma, kernel = kernel)
    clf = clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)
    print(confusion_matrix(test_Y, y_pred))
    print(classification_report(test_Y, y_pred))
    print("Accuracy:",metrics.accuracy_score(test_Y, y_pred))
    print("Time: ", time.time() - start_time) 

# SVM - Tuning for Chosen Hyperparameters 
def svm_tuning(c_list, gamma_list, train_X, train_Y):
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    train_X = scaler.fit_transform(train_X)
    param_grid = {'C': c_list, 'gamma': gamma_list}
    # param_grid_linear = {'C': [0.1,1, 10], 'gamma': [1,0.1,0.01,0.001],'kernel': [ 'linear' ]}
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
    grid.fit(train_X,train_Y.values.ravel())
    print(grid.best_estimator_)
    pd.set_option('display.max_rows', None)
    print(pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))

# Neural Network - Training and Tuning 

# NN Default Parameters 
def nn_model_default(train_X, train_Y, test_X, test_Y):
    # scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    # train_X = scaler.fit_transform(train_X)
    # test_X = scaler.fit_transform(test_X)
    clf = MLPClassifier()
    clf = clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)
    print("Accuracy:",metrics.accuracy_score(test_Y, y_pred))

# NN - Tuning for Chosen Hyperparameters 
def nn_tuning(param_grid,train_X, train_Y):
    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    grid = GridSearchCV(MLPClassifier(), param_grid=param_grid, n_jobs=-1, cv=3)
    grid.fit(train_X, train_Y.values.ravel())
    print(grid.best_estimator_)
    pd.set_option('display.max_rows', None)
    print(pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1))

# NN - Model w/ Best Hyperparameter Values
def nn_model(alpha, hidden_layers, learning_rate, train_X, train_Y, test_X, test_Y):
    start_time = time.time()
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.fit_transform(test_X)
    clf = MLPClassifier(alpha = alpha, hidden_layer_sizes = hidden_layers, learning_rate = learning_rate)
    clf = clf.fit(train_X, train_Y)
    y_pred = clf.predict(test_X)
    print(classification_report(test_Y, y_pred))
    print("Accuracy:",metrics.accuracy_score(test_Y, y_pred))
    print("Time: ", time.time() - start_time) 

# Plotting 

# plots classifier's hyperparmeter performance over range of values
def generate_plot(classifier,parameter, parameter_list, train_X, train_Y):
    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
    train_X = scaler.fit_transform(train_X)
    train_scoreNum, test_scoreNum = validation_curve(
                                    classifier,
                                    X = train_X, y = train_Y.values.ravel(), 
                                    param_name = parameter, 
                                    param_range = parameter_list , cv = 3)
    train_scores_mean = np.mean(train_scoreNum, axis=1)
    test_scores_mean = np.mean(test_scoreNum, axis=1)
    print(train_scores_mean)
    print(test_scores_mean)
    plt.title(classifier)
    plt.xlabel(parameter)
    plt.ylabel("Accuracy Score")
    plt.plot(parameter_list, train_scores_mean, label='Training Score', color='red',markerfacecolor='red')
    plt.plot(parameter_list, test_scores_mean, label='Testing Data',color='blue',markerfacecolor='blue')
    plt.legend(loc="best")
    plt.show()

# generates bar chart of accuracies on testing dataset for each classifier
def generate_bar():
    objects = ['Random Forest', 'SVM', 'Neural Network']
    y_pos = np.arange(len(objects))
    performance = [0.815,0.827,.821]
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('Accuracy')
    plt.title('Classifiers')
    plt.show()
    
# generates classifier's hyperparmeter performance over range of values in bar graph form 
def generate_barplot(classifier, parameter, parameter_list, train_X, train_Y): 
    scaler = MinMaxScaler()
    train_X = scaler.fit_transform(train_X)
    train_scoreNum, test_scoreNum = validation_curve(classifier,X=train_X, y=train_Y.values.ravel(), param_name=parameter,param_range=parameter_list, cv=3)
    width = 0.25
    train_bars = np.mean(train_scoreNum, axis=1)
    test_bars = np.mean(test_scoreNum, axis=1)
    r1 = np.arange(len(train_bars))
    r2 = [x + width for x in r1]
    plt.bar(r1, train_bars, width=width, edgecolor='white', label='Training')
    plt.bar(r2, test_bars, width=width, edgecolor='white', label='Testing')
    plt.xlabel(parameter)
    plt.xticks([r + (width/2) for r in range(len(train_bars))], parameter_list)
    plt.ylabel("Accuracy Score")
    plt.legend(loc="best")
    plt.show()
    
def main():
    credit_fraud = pd.read_csv("default_credit_card_clients.csv")
    X, Y = credit_fraud.iloc[:, :-1], credit_fraud.iloc[:, [-1]]
    train, test = train_test_split(credit_fraud, test_size=0.2)
    train_X, train_Y = train.iloc[:, :-1], train.iloc[:, [-1]]
    test_X, test_Y = test.iloc[:, :-1], test.iloc[:, [-1]]
    # print(train_X.shape)
    # print(train_Y.shape)

    # Default Random Forest w/ Bagging 
    # rf_model_default(train_X,train_Y,test_X,test_Y)

    # Random Forest - Grid Search for Best Hyperparameters  Time: 26.2min
    # n_estimators_list = [50, 100, 150, 200]
    # max_depth_list = [5, 10, 15, 25]
    # samples_split_list = [2, 5, 10,15]
    # leaf_list = [1, 2, 5] 
    # rf_tuning(n_estimators_list, max_depth_list, samples_split_list, leaf_list,train_X, train_Y)

    # Random Forest - Using Best Hyperparameters 
    # rf_model(200,10,15,2,train_X,train_Y,test_X,test_Y)

    # Default SVM 
    # svm_model_default(train_X, train_Y, test_X, test_Y)

    #SVM - Grid Search for Best Hyperparameters (Non-Linear Kernels) Time: ? 32 mins
    # C = [1/16,1/8,1/4,1/2,1,2,4,8,16]
    # gamma_list =  [1/16,1/8,1/4,1/2,1,2,4,8,16]
    # svm_tuning(C, gamma_list,train_X, train_Y)

    # SVM- Using Best Hyperparameters 
    # svm_model(8,1,'rbf',train_X,train_Y,test_X,test_Y)

    #  Generate Validation Curve - SVM 
    # generate_plot(SVC(), "gamma", gamma_list, train_X, train_Y)

    # Generate Bar Graph - SVM 
    # generate_barplot_svm("kernel",['rbf', 'poly', 'sigmoid'],train_X,train_Y)

    # Default Neural Network 
    # nn_model_default(train_X,train_Y,test_X,test_Y)
    # Neural Network Tuning 

    # param_grid = dict(learning_rate= ['constant', 'invscaling', 'adaptive'],alpha=[0.1, 0.01, 0.001, 0.0001], hidden_layer_sizes=[[20,20],[20,20,20], [30,30], [30,30,30], [40,40], [40,40,40], [50,50], [50, 50, 50]])    
    # nn_tuning(param_grid,train_X, train_Y)

    # Neural Network w/ Best Hyperparameters 
    # nn_model(.1,[50,50,50],"invscaling",train_X,train_Y,test_X,test_Y)
    # generate_barplot(SVC(),"kernel",['rbf', 'poly', 'sigmoid'],train_X,train_Y)

    # generate_bar()

if __name__ == '__main__':
	main()

