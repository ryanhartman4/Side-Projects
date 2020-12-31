#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 18:15:44 2020

@author: ryanhartman
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic_test = pd.read_excel('/Users/ryanhartman/Desktop/Graziadio/Fall 20/Second Half/DESC 620/Titanic Data/TitanicTrainingData.xls')
titanic_test = pd.DataFrame(titanic_test)
print(titanic_test.head(5))

print(titanic_test.columns.values)

titanic_original = titanic_test.copy()

titanic_test = titanic_test.drop(['PassengerId','Cabin'], axis = 1)
print("")
print(titanic_test.columns.values)


titanic_test = titanic_test.drop('Name', axis = 1)

titanic_test.describe()

titanic_test = titanic_test.drop('Ticket', axis = 1)

g = sns.FacetGrid(titanic_test, col='Survived')
g.map(plt.hist, 'Age', bins=20)

titanic_test['Sex'].replace(np.NaN,'female')
titanic_test['Sex'].replace('inf','female')

titanic_test.Sex.unique()

titanic_test.Pclass.unique()

#replace male, female with 0,1

titanic_test['Sex'] = titanic_test['Sex'].replace('female',1)
titanic_test['Sex'] = titanic_test['Sex'].replace('male',0)

#checking columns in df 

print(titanic_test.columns.values)

#engineering additional features based off of observations in SPSS
#Adding PclassXSex, FamilySize(SibSp+ParCh), AgeXSex, AgeXPclass,

titanic_test['PclassXSex'] = titanic_test['Pclass']*titanic_test['Sex']
titanic_test['FamilySize'] = titanic_test['SibSp']+titanic_test['Parch']
titanic_test['AgeXSex']= titanic_test['Age']*titanic_test['Sex']
titanic_test['AgeXPclass'] = titanic_test['Age'] * titanic_test['Pclass']

#creating categorical variable for Age. 0-15 = child, 16-30 = young adult, 30-60 = middle age, 60+ = senior
#child = 0, young adult = 1, middle aged = 2, senior = 3

def age_group(series):
    if series <= 15:
        return 0
    elif series <= 30:
        return 1
    elif series <= 60:
        return 2
    else:
        return 3
    
titanic_test['Age_Cat'] = titanic_test['Age'].apply(age_group)

print(titanic_test.columns.values)

titanic_test = titanic_test.drop('Embarked', axis = 1)

titanic_test['Survived']=titanic_test['Survived'].dropna()

titanic_test.isnull().values.any()
titanic_test['Age'].describe()
titanic_test['Age'] = titanic_test['Age'].fillna(10.20)
titanic_test['Age'] = titanic_test['Age'].replace(10.20,25.82)

titanic_test['AgeXPclass'] = titanic_test['AgeXPclass'].fillna(61.938151)


for column in titanic_test:
    if titanic_test[column].isnull().values.any():
        print(column)
    
titanic_test['AgeXSex'] = titanic_test['AgeXSex'].fillna(10.204)

for column in titanic_test:
    if titanic_test[column].isnull().values.any():
        print(column)
    else:
        print('None')
#creating logistic regression model using sci-kit learn module
#creating random forest model using sklearn module

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#creating train and test datasets
X_train = titanic_test.drop('Survived', axis = 1)
Y_train = titanic_test['Survived']
X_test = titanic_test.drop('Survived', axis = 1).copy()

logReg = LogisticRegression()

logReg.fit(X_train,Y_train)

accLogReg = round(logReg.score(X_train,Y_train)*100,2)
print('The accuracy of this model is {}%'.format(accLogReg))
print("")

logRegCoEff = pd.DataFrame(titanic_test.columns.delete(0))
logRegCoEff.columns = ['Feature']
logRegCoEff['logOddsCoefficient'] = pd.Series(logReg.coef_[0])
logRegCoEff.sort_values(by = 'logOddsCoefficient', ascending = False)

rf = RandomForestClassifier(n_estimators=400,max_samples = 500, max_features='sqrt', oob_score = True)

rf.fit(X_train,Y_train)

RFAcc = round(rf.score(X_train,Y_train),2)*100

print('The accuracy of the random forest model is {}%'.format(RFAcc))
#Determining which features held the  most importance in this model
RF_Imp_Feat = pd.DataFrame({'Varables':X_train.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending = False)

RF_Imp_Feat
RF_Imp_Feat['Importance'].sum()

X_train2 = X_train.drop('Fare', axis = 1)

rf.fit(X_train2,Y_train)

RFAcc = round(rf.score(X_train2,Y_train),2)*100

print('The accuracy of the random forest model is {}%'.format(RFAcc))
#Determining which features held the  most importance in this model
RF_Imp_Feat = pd.DataFrame({'Varables':X_train2.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending = False)

RF_Imp_Feat

X_train3 = X_train2.drop('Age_Cat', axis =1)
X_train3 = X_train3.drop('Parch', axis = 1)
X_train3 = X_train3.drop('SibSp', axis = 1)

rf.fit(X_train3,Y_train)

RFacc = round(rf.score(X_train3,Y_train),2)*100

print('The accuracy of this model is {}%'.format(RFacc))

RF_Imp_Feat = pd.DataFrame({'Variables':X_train3.columns,'Importances':rf.feature_importances_}).sort_values('Importances',ascending=False)

RF_Imp_Feat
