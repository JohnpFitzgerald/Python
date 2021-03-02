#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:53:36 2020

@author: farshad.toosi and Mohammed Hasanuzzaman
"""
# -*- coding: utf-8 -*-"""Repeat Exam 2020 Programming for Data Analytic """
###############################################
# Please write your name and student ID:
#John Fitzgerald
#r00156081
###############################################

#All questions attempted. I have set working folder as C:\CIT\ - it has the 
#input data file bank-full.csv and all output files that are created
#are written to this folder also   - JF

#import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.externals.six import StringIO
#from  sklearn.tree import export_graphviz
import pydotplus
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC

def cleanFile():
    #read the file and remove empty spaces from fields
     dataFile = pd.read_csv("c:\\CIT\\bank-full.csv")
     #clear file of empty spaces 
     dataFile.columns = dataFile.columns.str.strip()
     #remove columns will null 'nan' values and 'null' values
     dataFile=dataFile.dropna()
     dataFile = dataFile[dataFile.notnull().all(axis = 1)]
     #re classify the fields for  
     age_e = pd.Series(LabelEncoder().fit_transform(dataFile["age"]))
     marital_e = pd.Series(LabelEncoder().fit_transform(dataFile["marital"]))
     job_e = pd.Series(LabelEncoder().fit_transform(dataFile["job"]))
     education_e = pd.Series(LabelEncoder().fit_transform(dataFile["education"]))
     housing_e = pd.Series(LabelEncoder().fit_transform(dataFile["housing"]))
     poutcome_e = pd.Series(LabelEncoder().fit_transform(dataFile["poutcome"]))
     balance_e = pd.Series(LabelEncoder().fit_transform(dataFile["balance"]))     
     default_e = pd.Series(LabelEncoder().fit_transform(dataFile["default"]))
     loan_e = pd.Series(LabelEncoder().fit_transform(dataFile["loan"])) 
     y_e = pd.Series(LabelEncoder().fit_transform(dataFile["y"]))
     e=pd.concat([age_e,marital_e,job_e,education_e,housing_e,poutcome_e,balance_e, default_e, loan_e, y_e], axis=1)     
     e.columns = ['age','marital','job','education','housing','poutcome','balance','default','loan','y']
     return e

def task1():
    """
    Your implementation goes here.
    """
    #call the read file function
    e = cleanFile()
    
    #create 2 datasets from the cleaned file
    yDataset = e[['age','job','poutcome','balance','default','y']]
    loanDataset = e[['age','job','poutcome','balance','default','loan']] 
    
    #test display of datasets
    #print(loanDataset, yDataset)
    
    #split dataset in features and target variable
    # yDataset
    feature1 = yDataset[["age", "job","poutcome","balance","default"]] # Features of the 'y' dataset
    target1 = yDataset[["y"]] # Target variable for 'y' class attribute
    target1=target1.values
    
    # loanDataset
    feature2 = loanDataset[["age", "job","poutcome","balance","default"]] #features of the 'loan' dataset
    target2 = loanDataset[["loan"]] # target variable for the 'loan' class attribute
    target2=target2.values
    
    # Split dataset into training set and test set for both datasets
    X_train1, X_test1, y_train1, y_test1 = train_test_split(feature1, target1, test_size=0.1, random_state=1) # 90% training and 10% test
    X_train2, X_test2, y_train2, y_test2 = train_test_split(feature2, target2, test_size=0.1, random_state=1) # 90% training and 10% test
    
    # Create Decision Tree classifer object
    clft = DecisionTreeClassifier()
    clft.fit(X_train1, y_train1)
    clft.fit(X_train2, y_train2)
    yScores = model_selection.cross_val_score(clft, X_train1, y_train1, cv=10)
    loanScores = model_selection.cross_val_score(clft, X_train2, y_train2, cv=10)
    print("***********************************************")
    print("********** TASK 1 Outputs here ****************")
    print("***********************************************")    
    print("DataSet 1: ",yScores ,"\nDataset 2: ",loanScores)
    print("DataSet 1 average: ",yScores.mean(), " DataSet 1 Standard Deviation: ",yScores.std())
    print("DataSet 2 average: ",loanScores.mean(), " DataSet 2 Standard Deviation: ", loanScores.std())
    
    #test the classifiers
    predictTree=clft.predict(X_test1)
    print("Dataset1 accuracy score: ",metrics.accuracy_score(predictTree, y_test1))
    
    predictTree=clft.predict(X_test2)
    print("Dataset2 accuracy score: ",metrics.accuracy_score(predictTree, y_test2))
    #nested cross fold validation is best here - validate decision tree
    parameter_grid = { 'max_depth' : [1,2,3,4,5],
                      'max_features': [1,2,3,4],
                      'criterion': ["gini", "entropy"]}
    #specify cross validation for dataset 1 
    cross_validation = StratifiedKFold(n_splits=10)
    cross_validation.get_n_splits(X_train1,y_train1)
    #find optimised parameters
    gsclft = GridSearchCV(clft, param_grid = parameter_grid,
                               cv = cross_validation)
    #best estimator is fitted for X_train, y_train (1 and 2)
    gsclft.fit(X_train1, y_train1)
    #outer loop with cv 5
    scoresclft = model_selection.cross_val_score(gsclft, X_train1, y_train1, cv=5)
    #print(scoresclft.shape)
    print("Dataset 1 Nested cross fold accuracy is/Mean Score: ", np.mean(scoresclft))
    print("1. Best Score: {}".format(gsclft.best_score_))
    print("1. Best params: {}".format(gsclft.best_params_))
    #make best tree with parameters suggested by CV (above)
    besttree= tree.DecisionTreeClassifier(
             criterion='entropy', max_depth=4, max_features=3)
    besttree.fit(X_train1, y_train1)

    #visualisation for dataset 1
    tree.export_graphviz(besttree,out_file='Task1-DecisionTreeBest1.dot',
                         feature_names=X_train1.columns.values)
    dot_data = StringIO()
    # use graphviz to view the dot file
    tree.export_graphviz(besttree, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    #create a PDF of the tree graph in the working folder
    graph.write_pdf("task1-tree1.pdf")
  

    #specify cross validation for dataset 2 
    cross_validation = StratifiedKFold(n_splits=10)
    cross_validation.get_n_splits(X_train2,y_train2)
    #find optimised parameters
    gsclft = GridSearchCV(clft, param_grid = parameter_grid,
                               cv = cross_validation)
    #best estimator is fitted for X_train, y_train (1 and 2)
    gsclft.fit(X_train2, y_train2)
    #outer loop with cv 5
    scoresclft = model_selection.cross_val_score(gsclft, X_train2, y_train2, cv=5)
    #print(scoresclft.shape)
    print("Dataset 2 Nested cross fold accuracy is/Mean Score: ", np.mean(scoresclft))
    print("2. Best Score: {}".format(gsclft.best_score_))
    print("2. Best params: {}".format(gsclft.best_params_))
    #make best tree with parameters suggested by CV (above)
    besttree= tree.DecisionTreeClassifier(
             criterion='entropy', max_depth=4, max_features=3)
    besttree.fit(X_train2, y_train2)

    #visualisation for dataset 2
    tree.export_graphviz(besttree,out_file='Task1-DecisionTreeBest2.dot',
                         feature_names=X_train1.columns.values)
    dot_data = StringIO()
    # use graphviz to view the dot file
    tree.export_graphviz(besttree, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("task1-tree2.pdf")

    #On my last run ..these were the results:

# ***********************************************
# ********** TASK 1 Outputs here ****************
# ***********************************************
# DataSet 1:  [0.8289506  0.81617105 0.826493   0.82256083 0.82059474 0.82305235
#  0.83017941 0.81248464 0.82231507 0.82300885] 
# Dataset 2:  [0.76480708 0.76308675 0.75301057 0.76677316 0.76112067 0.7571885
#  0.75669698 0.75743426 0.75522241 0.76204523]
# DataSet 1 average:  0.8225810520729802  DataSet 1 Standard Deviation:  0.0051093721252694424
# DataSet 2 average:  0.7597385609543149  DataSet 2 Standard Deviation:  0.004246671300132212
# Dataset1 accuracy score:  0.7518796992481203
# Dataset2 accuracy score:  0.7669172932330827
# Dataset 1 Nested cross fold accuracy is/Mean Score:  0.893116047552945
# 1. Best Score: 0.8930669524932864
# 1. Best params: {'criterion': 'gini', 'max_depth': 5, 'max_features': 4}
# Dataset 2 Nested cross fold accuracy is/Mean Score:  0.8394651853656416
# 2. Best Score: 0.8398092829855109
# 2. Best params: {'criterion': 'gini', 'max_depth': 2, 'max_features': 1}    

#From the results above I would say that DataSet 2 shows better accuracy under the 
# the train data and when testing the classifiers    
def task2():
    """
    Your implementation goes here.
    """
    #call the read file function
    e = cleanFile()
    
    #create dataset from the cleaned file
    df = e[['age','balance']]
    #print(df)
    #calling the k-means algorithm (unsupervised learning algorithm) 
    #I am creating 8 clusters of balance (y axis) against age (x axis)
    kmeans = KMeans(n_clusters=8).fit(df)
    centroids = kmeans.cluster_centers_
    #show the mean values for age and balance
    print("***********************************************")
    print("********** TASK 2 Outputs here ****************")
    print("***********************************************")
    print(centroids)
    plt.title("Task2: Scatter plot diagram of Account Balance by Age",fontweight="bold")
    plt.xlabel('Age')
    plt.ylabel("Balance")
    #create scatter diagram
    plt.scatter(df['age'], df['balance'], c= kmeans.labels_.astype(float), s=50, alpha=0.5)
    #red dot on graph showing mean value for each cluster
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
    plt.savefig('task2-ScatterPlotofAgeandBal.png')
    plt.show()

#For task 2 I applied KMeans learning algorithm with 8 clusters Balance figures.
#this is the optimal number of clusters to view the balances by age grouping. The red
#dot in each cluster shows the mean for amout for that age.     

#I have commmented the next function out as I could not get it to work proerly to 
#perform cross validation on all the algorithms used for the task.
    
# def task3crossValidation(model,X_train, y_train):
#     parameter_grid = { 'max_depth' : [1,2,3,4,5],
#                        'max_features': [1,2,3,4],
#                        'criterion': ["gini", "entropy"]}
#     cross_validation = StratifiedKFold(n_splits=10)
#     cross_validation.get_n_splits(X_train,y_train)
#     gsclft = GridSearchCV(model, param_grid = parameter_grid,
#                                cv = cross_validation)   
#     gsclft.fit(X_train, y_train)
#     scoresclft = model_selection.cross_val_score(gsclft, X_train, y_train, cv=5)
#     #print(scoresclft.shape)
#     print("Nested cross fold accuracy is/Mean Score: ", np.mean(scoresclft))
#     print("Best Score: {}".format(gsclft.best_score_))
#     print("Best params: {}".format(gsclft.best_params_))
    
def task3():
    """
    Your implementation goes here.
    """
    #call the read file function
    e = cleanFile()
    
    #create dataset from the cleaned file
    df = e[['age','job','education','loan','y']]
    #print(df)    
    print("***********************************************")
    print("********** TASK 3 Outputs here ****************")
    print("***********************************************")
    feature = df[["age", "job","education","loan"]] # Featuresdataset
    target =  df[["y"]] # Target variable for 'y' class attribute
    target=target.values
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.1, random_state=55) # 90% training and 10% test 
    
    #KNeighborsClassifier
    #use ravel method here to mitigate against column-vector error
    target = y_train.ravel()
    y_train = np.array(target).astype(int)
    
    knn = KNeighborsClassifier(n_neighbors=1)   
    knn.fit(X_train, y_train) 
    #scores = model_selection.cross_val_score(knn, X_train,y_train,cv=10)
    predictTree=knn.predict(X_test)
    print("KNeighborsClassifier - Dataset accuracy score: ",metrics.accuracy_score(predictTree, y_test))
    #task3crossValidation(knn,X_train,y_train)
    
    #DecisionTreeClassifier
    clft = DecisionTreeClassifier()
    clft.fit(X_train, y_train)
    predictTree=clft.predict(X_test)
    print("DecisionTreeClassifier - Dataset accuracy score: ",metrics.accuracy_score(predictTree, y_test))
    #task3crossValidation(clft,X_train,y_train)
    
    #RandomForestClassifier
    clff = RandomForestClassifier()
    clff.fit(X_train, y_train)
    #scores = model_selection.cross_val_score(clff, X_train,y_train,cv=10)
    #print("RandomForestClassifer -Dataset accuracy score: ",scores)
    predictTree=clff.predict(X_test)
    print("RandomForestClassifer - Dataset accuracy score: ",metrics.accuracy_score(predictTree, y_test))
    #task3crossValidation(clff,X_train,y_train) 
    
    #GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    #scores = model_selection.cross_val_score(gnb, X_train,y_train,cv=10)
    #print("GaussianNB -Dataset accuracy score: ",scores)    
    predictTree=gnb.predict(X_test)
    print("GaussianNB - Dataset accuracy score: ",metrics.accuracy_score(predictTree, y_test))
    #task3crossValidation(gnb,X_train,y_train)
   
    
    #SVM
    svb = SVC(C=1E-01, kernel='rbf', gamma=0.1)
    svb.fit(X_train, y_train)
    #scores = model_selection.cross_val_score(clf, X_train,y_train,cv=10)
    #print("SVM -Dataset accuracy score: ",scores)    
    predictTree=svb.predict(X_test)
    print("SVM  - Dataset accuracy score: ",metrics.accuracy_score(predictTree, y_test))
    #task3crossValidation(svb,X_train,y_train)
    
    #Task 3 shows test data as 10% of the whole data. From the results received - SVM 
    #is consistently the most accurate 
# ***********************************************
# ********** TASK 3 Outputs here ****************
# ***********************************************
# KNeighborsClassifier - Dataset accuracy score:  0.7985404688191066
# DecisionTreeClassifier - Dataset accuracy score:  0.8834586466165414
# RandomForestClassifer - Dataset accuracy score:  0.8819106590004423
# GaussianNB - Dataset accuracy score:  0.8839009287925697
# SVM  - Dataset accuracy score:  0.8874391862007961    
def task4():
    """
    Your implementation goes here.
    """
    #call the read file function
    e = cleanFile()
    
    #create dataset from the cleaned file
    df = e[['housing','balance','y']]
    #print(df)
    print("***********************************************")
    print("********** TASK 4 Outputs here ****************")
    print("***********************************************")   
    feature = df[["housing", "balance"]] # Featuresdataset
    target =  df[["y"]] # Target variable for 'y' class attribute
    target=target.values
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=100) # 80% training and 10% test 

    # Create Decision Tree classifer object
    clft = tree.DecisionTreeClassifier()
    clft.fit(X_train, y_train)
    Scores = model_selection.cross_val_score(clft, X_train, y_train, cv=10)
    print("DataSet4: ",Scores )
    print("DataSet4 average: ",Scores.mean(), " DataSet4 Standard Deviation: ",Scores.std())
    
    #test the classifiers
    predictTree=clft.predict(X_test)
    print("Dataset4 accuracy score: ",metrics.accuracy_score(predictTree, y_test))
    
    #nested cross fold validation is best here - validate decision tree
    parameter_grid = { 'max_depth' : [1,2,3,4,5],
                      'max_features': [1,2],
                      'criterion': ["gini", "entropy"]}
    #specify cross validation for dataset4 
    cross_validation = StratifiedKFold(n_splits=10)
    cross_validation.get_n_splits(X_train,y_train)
    #find optimised parameters
    gsclft = GridSearchCV(clft, param_grid = parameter_grid,
                               cv = cross_validation)
    #best estimator is fitted for X_train, y_train 
    gsclft.fit(X_train, y_train)
    #outer loop with cv 5
    scoresclft = model_selection.cross_val_score(gsclft, X_train, y_train, cv=5)
    print(scoresclft.shape)
    print("Dataset4 -Nested cross fold accuracy is/Mean Score: ", np.mean(scoresclft))
    print("Best Score: {}".format(gsclft.best_score_))
    print("Best params: {}".format(gsclft.best_params_))
    #make best tree with parameters suggested by CV (above)
    besttree= tree.DecisionTreeClassifier(
             criterion='entropy', max_depth=5, max_features=2)
    besttree.fit(X_train, y_train)

    #visualisation for dataset 1
    tree.export_graphviz(besttree,out_file='Task4-DecisionTreeBest.dot',
                         feature_names=X_train.columns.values)
    dot_data = StringIO()
    # use graphviz to view the dot file
    tree.export_graphviz(besttree, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("task4-tree.pdf")
 #   Image(graph.create_png())

# ***********************************************
# ********** TASK 4 Outputs here ****************
# ***********************************************
# DataSet4:  [0.86839923 0.86342273 0.85623445 0.86701686 0.86756981 0.86204036
#  0.86286978 0.86784628 0.85923673 0.8642146 ]
# DataSet4 average:  0.8638850829783641  DataSet4 Standard Deviation:  0.003799334904468888
# Dataset4 accuracy score:  0.8680747539533341
# (5,)
# Dataset4 -Nested cross fold accuracy is/Mean Score:  0.8824928107859407
# Best Score: 0.8824928098874294
# Best params: {'criterion': 'gini', 'max_depth': 1, 'max_features': 1}

def task5():
    """
    Your implementation goes here.
    """
    dataFile = pd.read_csv("c:\\CIT\\bank-full.csv")
    #clear file of empty spaces 
    dataFile.columns = dataFile.columns.str.strip()
    #remove columns will null 'nan' values and 'null' values
    dataFile=dataFile.dropna()
    dataFile = dataFile[dataFile.notnull().all(axis = 1)]
    print("***********************************************")
    print("********** TASK 5 Outputs here ****************")
    print("***********************************************")    
    series = dataFile["marital"].value_counts()
    series.plot(kind='bar', figsize=[9,6])
    plt.xlabel("Marital Status recorded")
    plt.ylabel("No of entries")
    plt.title("Task5: Martial Status for all entries",fontweight="bold")
    plt.savefig('task5-MaritalStatus.png')
    plt.show() 
    # 3types of marital status. Large groups are married (over 25000),Single
    #people are approx. half the vale of married and Divorced are jus tover 5000.
    
    
def task6():
    """
    Your implementation goes here.
    """
    dataFile = pd.read_csv("c:\\CIT\\bank-full.csv")
    #clear file of empty spaces 
    dataFile.columns = dataFile.columns.str.strip()
    #remove columns will null 'nan' values and 'null' values
    dataFile=dataFile.dropna()
    dataFile = dataFile[dataFile.notnull().all(axis = 1)]
    print("***********************************************")
    print("********** TASK 6 Outputs here ****************")
    print("***********************************************")
    import seaborn as sns
    sns.scatterplot(x='age', y='balance',hue='age', data=dataFile)  
    plt.title("Task6: Age and Bank Balance",fontweight="bold")
    plt.savefig('task6-scatterofBalances.png')
    plt.show()
    #The age range 50-60 shows that highest volume of valances on
    # the scatter. 50-60 has highest in range 4000-6000, 6000-8000 and
    #over 10,000 on the plot.

def task7():
    """
    Your implementation goes here.
    """
    dataFile = pd.read_csv("c:\\CIT\\bank-full.csv")
    #clear file of empty spaces 
    dataFile.columns = dataFile.columns.str.strip()
    #remove columns will null 'nan' values and 'null' values
    dataFile=dataFile.dropna()
    dataFile = dataFile[dataFile.notnull().all(axis = 1)]
    print("***********************************************")
    print("********** TASK 7 Outputs here ****************")
    print("***********************************************")
    import seaborn as sns
    sns.set_style('whitegrid')
    sns.boxplot(y='age',data=dataFile)
    plt.title("Task7: Boxplot of Age attribute",fontweight="bold")
    plt.savefig('task7-BoxplotAge.png')
    plt.show()
    #median age is just under 40 according to the graph. lowest age range is 18 (approx.)
    # and highest is 70. Outliers are ages 70+ in this diagram with ages recorded up to 120 
    # in the file (some of the higher figures should probably be checked for validity)
    
    
    
#cleanFile()    
task1()    
task2()    
task3()
task4()
task5()
task6()
task7()
