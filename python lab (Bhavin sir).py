# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 20:03:29 2023

@author: HP
"""
#import all packages 
import pandas as pd 
import matplotlib.pyplot as mlt
import numpy as np
import seaborn as sns
#import data set
data=pd.read_csv(r"C:\Users\HP\Downloads\winequality-red.csv")
#performing basic EDA 
pd.set_option('display.max_columns',40)
data.head()
data.head(10)
data.columns
data.tail()
data.shape
data.info()
data.isna()
data.describe()
data.isna().sum()
data1=pd.read_csv(r"C:\Users\HP\Downloads\winequality-red.csv",na_values=[0])
data1.isna().sum()
#replacing with null values
mean=data["citric acid"].mean()
data['citric acid']=data["citric acid"].replace(0,mean)
data.isna().sum()
#import all packages 
import pandas as pd 
import matplotlib.pyplot as mlt
import numpy as np
import seaborn as sns
#import data set
data=pd.read_csv(r"C:\Users\HP\Downloads\winequality-red.csv")
#performing basic EDA 
pd.set_option('display.max_columns',40)
data.head()
data.head(10)
data.columns
data.tail()
data.shape
data.info()
data.isna()
data.describe()
data.isna().sum()
data1=pd.read_csv(r"C:\Users\HP\Downloads\winequality-red.csv",na_values=[0])
data1.isna().sum()
#replacing with null values
mean=data["citric acid"].mean()
data['citric acid']=data["citric acid"].replace(0,mean)
data.isna().sum()
data.describe()
data.info()
#find corr and adapting vlaues
mlt.figure(figsize=(10,8))
sns.pairplot(data)
#ploting the graphs
mlt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)

columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol',]
for i in columns:
    mlt.figure(figsize=(10,8))
    sns.boxplot(x=data['quality'],y=data[i])
    
mlt.figure(figsize=(10,8))    
mlt.scatter(x=data['alcohol'],y=data['quality'])    
mlt.scatter(x=data['sulphates'],y=data['quality'])
#segregating input and output
x=data.drop(['quality'],axis=1)
y=data['quality']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20
                                               ,random_state=0)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
#####################
from sklearn import metrics
metrics.confusion_matrix(y_test,y_pred)
metrics.accuracy_score(y_test,y_pred)

print(metrics.classification_report(y_test,y_pred))

