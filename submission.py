#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:53:36 2020

@author: farshad.toosi and Mohammed Hasanuzzaman
"""
# -*- coding: utf-8 -*-"""Repeat Exam 2020 Programming for Data Analytic """
# Please write your name and student ID:
#John Fitzgerald
#r00156081
#import libraries
import pandas as pd
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

def cleanFile():
    #read the file and remove empty spaces from fields
     dataFile = pd.read_csv("bank-full.csv",skipinitialspace=True,na_values={''},keep_default_na=False)
     #clear file of empty spaces 
     dataFile.columns = dataFile.columns.str.strip()
     #filter out rows with missing values
     dataFile.dropna()
     #remove columns will null 'nan' values
     dataFile = dataFile[dataFile.notnull().all(axis = 1)]
     #return file for use in 12 questions
     return dataFile
def Task1():
    """
    Your implementation goes here.
    """
     #call the read file function
     dataFile = cleanFile()
     #set labels for single instance of each worktype
     labels = dataFile['workclass'].unique()
     #count of men group by workclass
     men =  dataFile['workclass'][dataFile['sex']=='Male'].value_counts().values
     #count of women grouped by workclass
     women = dataFile['workclass'][dataFile['sex']=='Female'].value_counts().values    
    
def Task2():
    """
    Your implementation goes here.
    """
    
def Task3():
    """
    Your implementation goes here.
    """
    
def Task4():
    """
    Your implementation goes here.
    """
    
def Task5():
    """
    Your implementation goes here.
    """
    
def Task6():
    """
    Your implementation goes here.
    """
    
def Task7():
    """
    Your implementation goes here.
    """
    
    
    

