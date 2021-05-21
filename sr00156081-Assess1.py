#!/usr/bin/env python3
# -*- coding: utf-8 -*-"""Second assessment Programming for Data Analytic """
# Please write your name and student ID:
#John Fitzgerald

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def cleanFile():
    #read the file and remove empty spaces from fields
     dataFile = pd.read_csv("adult.csv",skipinitialspace=True,na_values={''},keep_default_na=False)
     #clear file of empty spaces 
     dataFile.columns = dataFile.columns.str.strip()
     #filter out rows with missing values
     dataFile.dropna()
     #remove columns will null 'nan' values
     dataFile = dataFile[dataFile.notnull().all(axis = 1)]
     #return file for use in 12 questions
     return dataFile

def task1():
     #your implementation for task 1 goes here
     #call the read file function
     dataFile = cleanFile()
     #set labels for single instance of each worktype
     labels = dataFile['workclass'].unique()
     #count of men group by workclass
     men =  dataFile['workclass'][dataFile['sex']=='Male'].value_counts().values
     #count of women grouped by workclass
     women = dataFile['workclass'][dataFile['sex']=='Female'].value_counts().values
     #print(men)
     #print(women)
     #print(labels)
     #arrange labels for x axis  
     x = np.arange(len(labels)) 
     #width of bars
     width = 0.4  
     #set graph and size it
     fig, ax = plt.subplots(figsize=(12, 8))
     #set bars for male and female     
     bar1 = ax.bar(x, men, width, label='Male')
     bar2 = ax.bar(x + width, women, width, label='Female')
     plt.xticks(rotation=90)
     #Add labels and , title and headings to indicate results
     ax.set_ylabel('No. Of People')
     ax.set_xlabel('Work Type')     
     ax.set_title('Task 1: People in each WorkClass by Gender',fontweight="bold")
     ax.set_xticks(x)
     ax.set_xticklabels(labels)
     #show male and female descriptor
     ax.legend()
     #can use scalar formatting to show the lesser values on the graph
     #plt.yscale("log")
     #plt.gca().yaxis.set_major_formatter(ScalarFormatter())
     def autolabel(rects, xpos='center'):
         ha = {'center': 'center', 'right': 'left', 'left': 'right'}
         offset = {'center': 0, 'right': 1, 'left': -1}
         for rect in rects:
             height = rect.get_height()
             ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos]*3, 3), 
                        textcoords="offset points",  
                        ha=ha[xpos], va='bottom',size=10)
     autolabel(bar1, "center")
     autolabel(bar2, "center")     
     fig.tight_layout()
     plt.show()
     #comment for task1:

def task2():
     #your implementation for task 2 goes here
     dataFile = cleanFile()
     #labels = dataFile['education'].unique()
     #hrs = dataFile['education'][dataFile['hours-per-week']].value_counts().values
     #groupHrs = dataFile.groupby('hours-per-week')
     #print(groupHrs)
     #women = dataFile['education'][dataFile['sex']=='Female'].value_counts().values
     #print(hrs)
     Female = dataFile[dataFile['sex']=='Female']
     Female.groupby('education')['hours-per-week'].mean().plot(kind='bar')
     plt.title("Task2: Average hours worked by Education, Females",fontweight="bold")
     plt.xlabel('Education level')
     plt.ylabel("Average weekly hours")
     plt.xticks(rotation=90)
     plt.show()
     #comment for task2:
     #I have gone for the average hours worked for this task. not sure 
     #if that is what is required - it gave me more manageable results
     #you can see see the work patterns associated by the different levels
     #of eduation 


def task3():
     #your implementation for task 3 goes here
     dataFile = cleanFile() 
     #Country = dataFile[['native-country']]
     #groupCountries = dataFile.groupby('native-country')
     #print(groupCountries.count())
     dataFile['native-country'].value_counts().head(1).plot(kind='bar')
     plt.title("Task3: Country with maximum entry",fontweight="bold")
     plt.xlabel('Country',fontweight="bold")
     plt.ylabel("No. occurrences",fontweight="bold")
     plt.xticks(rotation=0)
     #plt.yscale("log")
     #plt.gca().yaxis.set_major_formatter(ScalarFormatter())
     plt.show() 
     #comment for task3:
     # simple bar chart sowing the country with maximum entries
     # on the file           

def task4():
    #your implementation for task 4 goes here    
     dataFile = cleanFile() 
     top5 = dataFile['native-country'].value_counts().head(5)
     secondThird = top5[1:3]
     secondThird.plot(kind='bar')
     plt.title("Task4: Next 2 maximum entries ",fontweight="bold")
     plt.xlabel('Country',fontweight="bold")
     plt.ylabel("No. occurrences",fontweight="bold")
     plt.xticks(rotation=0)
     plt.show()
     #comment for task4:
     #this bar chart shows the second, third and fourth countries with maximum
     #entries
def task5():
    #your implementation for task 5 goes here
    dataFile = cleanFile()
    dataFile.groupby('age')['hours-per-week'].mean().plot(kind='bar', figsize=[16,6])
    plt.xlabel('Age')
    plt.ylabel("Hours")
    plt.title("Task5: relationship between age and working hours",fontweight="bold")
    plt.show()
    #comment for task5:
    #from the graph - Ages 28-60 has the highest working hours per week

def task6():
    #your implementation for task 6 goes here
    dataFile = cleanFile()    
    dataFile.groupby('education')['age'].count().plot(kind='bar', figsize=[8,3])    
    plt.xlabel('Education level completed')
    plt.ylabel("age")
    plt.title("Task 6: Analyse education levels",fontweight="bold")
    plt.xticks(rotation=90)
    plt.show()
    #part 2 showing boxplot of education
    dataFile = dataFile[['education-num']]
    y= 'education-num'
    plt.figure(figsize=[7,7])
    sns.boxplot(y=y, data = dataFile)
    plt.xlabel('Education Levels')
    plt.ylabel("Education number")
    plt.title("Task 6: Analyse education levels Boxplot",fontweight="bold")
    plt.show()
    #comment for task6:
    #The outliers here appear to be values 1-4 corresponding 
    #to pre-school up to 7th-8th grade.
    #median appears to be number 10 which is 'Some College'


def task7():
     #your implementation for task 7 goes here
     dataFile = cleanFile()     
     fig, ax = plt.subplots()
     dataFile['Income'] = dataFile['Income'].str.replace(r'.', '')
     labels = dataFile['Income'].unique()
     men =  dataFile['Income'][dataFile['sex']=='Male'].value_counts().values
     women = dataFile['Income'][dataFile['sex']=='Female'].value_counts().values
     x = np.arange(len(labels))
     width = 0.35  
     bar1 = ax.bar(x, men, width, label='Male')
     bar2 = ax.bar(x + width, women, width, label='Female')
     ax.set_xlabel('Income levels') 
     ax.set_ylabel('No. of Persons')
     ax.set_title('Task 7:  Income levels by Gender',fontweight="bold")
     ax.set_xticks(x)
     ax.set_xticklabels(labels)
     ax.legend()
     def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
     autolabel(bar1)
     autolabel(bar2)
     fig.tight_layout()
     plt.show()
     #comment for task7:
     # income levels broken down by gender - disparity between wages 
     # in clearly visible in this graph

def task8():
     #your implementation for task 8 goes here
     dataFile = cleanFile()
     labels = dataFile['education'].unique()
     m1 = dataFile['education'][dataFile['marital-status']=='Divorced'].value_counts().values
     m2 = dataFile['education'][dataFile['marital-status']=='Married-AF-spouse'].value_counts().reindex(dataFile['education'].unique(), fill_value=0).values
     m3 = dataFile['education'][dataFile['marital-status']=='Married-civ-spouse'].value_counts().reindex(dataFile['education'].unique(), fill_value=0).values
     m4 = dataFile['education'][dataFile['marital-status']=='Married-spouse-absent'].value_counts().reindex(dataFile['education'].unique(), fill_value=0).values
     m5 = dataFile['education'][dataFile['marital-status']=='Never-married'].value_counts().reindex(dataFile['education'].unique(), fill_value=0).values
     m6 = dataFile['education'][dataFile['marital-status']=='Separated'].value_counts().reindex(dataFile['education'].unique(), fill_value=0).values
     m7 = dataFile['education'][dataFile['marital-status']=='Widowed'].value_counts().reindex(dataFile['education'].unique(), fill_value=0).values
     x = np.arange(len(labels))      
     dataFile = pd.DataFrame({'Divorced': m1,
                   'Married-AF-spouse': m2,
                   'Married-civ-spouse': m3,
                   'Married-spouse-absent': m4,
                   'Never-married': m5,
                   'Separated': m6,
                   'Widowed': m7
                   }, index=labels)
     ax = dataFile.plot.bar(rot=0, figsize=(8, 8))
     plt.xticks(rotation=90)
     ax.set_ylabel('No of People')
     ax.set_title('Task 8: Education and martial status',fontweight="bold")
     ax.legend()
     plt.show()
     #comment for task8:


def task9():
     #your implementation for task 9 goes here
     dataFile = cleanFile()     
     labels = dataFile['occupation'].unique()
     m1 = dataFile['occupation'][dataFile['marital-status']=='Divorced'].value_counts().reindex(dataFile['occupation'].unique(), fill_value=0).values
     m2 = dataFile['occupation'][dataFile['marital-status']=='Married-AF-spouse'].value_counts().reindex(dataFile['occupation'].unique(), fill_value=0).values
     m3 = dataFile['occupation'][dataFile['marital-status']=='Married-civ-spouse'].value_counts().reindex(dataFile['occupation'].unique(), fill_value=0).values
     m4 = dataFile['occupation'][dataFile['marital-status']=='Married-spouse-absent'].value_counts().reindex(dataFile['occupation'].unique(), fill_value=0).values
     m5 = dataFile['occupation'][dataFile['marital-status']=='Never-married'].value_counts().reindex(dataFile['occupation'].unique(), fill_value=0).values
     m6 = dataFile['occupation'][dataFile['marital-status']=='Separated'].value_counts().reindex(dataFile['occupation'].unique(), fill_value=0).values
     m7 = dataFile['occupation'][dataFile['marital-status']=='Widowed'].value_counts().reindex(dataFile['occupation'].unique(), fill_value=0).values
     x = np.arange(len(labels)) 
     #fig, ax = plt.subplots()
     df = pd.DataFrame({'Divorced': m1,
                   'Married-AF-spouse': m2,
                   'Married-civ-spouse': m3,
                   'Married-spouse-absent': m4,
                   'Never-married': m5,
                   'Separated': m6,
                   'Widowed': m7
                   }, index=labels)
     ax = df.plot.bar(rot=0, figsize=(8,8))
     plt.xticks(rotation=90)
     ax.set_ylabel('no. of people')
     plt.title("Task 9: Occupation and Marital Status",fontweight="bold")
     ax.set_xticks(x)
     ax.set_xticklabels(labels)
     ax.legend()
     plt.show()
     #comment for task9: Occupation by marital status 
     
def task10():
     #your implementation for task 10 goes here
     dataFile = cleanFile()   
     labels = dataFile['occupation'].unique()
     men =  dataFile['occupation'][dataFile['education']=='Bachelors'].value_counts().values
     women = dataFile['occupation'][dataFile['education']=='Masters'].value_counts().values
     x = np.arange(len(labels))
     width = 0.35
     fig, ax = plt.subplots(figsize=(9,10))
     bar1 = ax.bar(x, men, width, label='Bachelor')
     bar2 = ax.bar(x + width, women, width, label='Master')
     ax.set_xticks(x)
     ax.set_xticklabels(labels)
     ax.set_ylabel(' Number of People')    
     ax.set_title('Task 10: Work class of Bachelor and Masters',fontweight="bold")
     def autolabel(rects, xpos='center'):
         ha = {'center': 'center', 'right': 'left', 'left': 'right'}
         offset = {'center': 0, 'right': 1, 'left': -1}
         for rect in rects:
             height = rect.get_height()
             ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(offset[xpos]*3, 3), 
                        textcoords="offset points",  
                        ha=ha[xpos], va='bottom',size=10)
     autolabel(bar1, "center")
     autolabel(bar2, "center") 
     plt.xticks(rotation=90)
     plt.xlabel('Job type') 
     plt.legend()
     plt.show()
    
     #comment for task10: Male and females with bachelor and master degrees
     # and the class of work they are employed in
def task11():
     #your implementation for task 11 goes here
     dataFile = cleanFile()   
     labels = dataFile['marital-status'].unique()
     m1 = dataFile['marital-status'][dataFile['workclass']=='Private'].value_counts().sort_values(ascending=[False]).reindex(dataFile['marital-status'].unique(), fill_value=0).values
     m2 = dataFile['marital-status'][dataFile['workclass']=='State-gov'].value_counts().sort_values(ascending=[False]).reindex(dataFile['marital-status'].unique(), fill_value=0).values
     m3 = dataFile['marital-status'][dataFile['workclass']=='Self-emp-not-inc'].value_counts().sort_values(ascending=[False]).reindex(dataFile['marital-status'].unique(), fill_value=0).values
     m4 = dataFile['marital-status'][dataFile['workclass']=='Local-gov'].value_counts().sort_values(ascending=[False]).reindex(dataFile['marital-status'].unique(), fill_value=0).values
     m5 = dataFile['marital-status'][dataFile['workclass']=='Federal-gov'].value_counts().sort_values(ascending=[False]).reindex(dataFile['marital-status'].unique(), fill_value=0).values
     m6 = dataFile['marital-status'][dataFile['workclass']=='Self-emp-inc'].value_counts().sort_values(ascending=[False]).reindex(dataFile['marital-status'].unique(), fill_value=0).values
     x = np.arange(len(labels))
     df = pd.DataFrame({'Private': m1,
                   'State-gov': m2,
                   'Self-emp-not-inc': m3,
                   'Local-gov': m4,
                   'Federal-gov': m5,
                   'Self-emp-inc': m6,
                   }, index=labels)
     ax = df.plot.line(rot=0, figsize=(8,8))
     plt.xticks(rotation=90)
     ax.set_ylabel('Counts') 
     plt.title("Task 11: Maritals status as private work classes.",fontweight="bold")
     ax.set_xticks(x) #ax
     ax.set_xticklabels(labels) 
     plt.legend() 
     plt.show()
     #comment for task11: grouping marital status by workclass. Those in the 
     #private sector are consistently higher across all marital types

def task12():
     #your implementation for task 12 goes here
    #part 2 showing boxplot of education
     dataFile = cleanFile() 
     #labels = dataFile['marital-status'].unique()
    #ax = dataFile.plot.bar(rot=0, figsize=(15, 6))
     x= 'marital-status'
     y= 'education-num'
     plt.figure(figsize=[7,7])
     plt.xticks(rotation=90)
     sns.boxplot(x=x, y=y, data = dataFile)
     plt.xlabel('marital status')
     plt.ylabel("Education number")
     plt.title("Task 12: Analysis of Marital status against education level",fontweight="bold")
     plt.show()
    
     #comment for task12: I am showing in this graph the outliers  of marital
     #status paired against education numbers  
     
     
     
     
     
task1()     
task2()  
task3()
task4()
task5() 
task6()
task7()
task8()
task9()
task10()
task11()
task12()    
