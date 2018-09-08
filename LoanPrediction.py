
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


trainData = pd.read_csv('H:/train.csv')

trainData.head(2)
trainData.shape
trainData.isnull().sum()
trainData.columns

#Handling with missing data

trainData.Gender.fillna(trainData.Gender.max(),inplace =True)
trainData.Married.fillna(trainData.Married.max(),inplace=True)
trainData.Credit_History.fillna(trainData.Credit_History.max(),inplace=True)
trainData.LoanAmount.fillna(trainData.LoanAmount.mean(),inplace=True)
trainData.Loan_Amount_Term.fillna(trainData.Loan_Amount_Term.mean(),inplace=True)
trainData.Self_Employed.fillna(trainData.Self_Employed.max(),inplace=True)
trainData.Dependents.fillna(0,inplace=True)

#Convert string values to numerical values because to algorithm can understand only numerical value not string values

trainData.Gender.value_counts()
gender_cat = pd.get_dummies(trainData.Gender,prefix='gender').gender_Female
trainData.Married.value_counts()
married_category = pd.get_dummies(trainData.Married,prefix='marriage').marriage_Yes
trainData.Education.value_counts()
graduate_category = pd.get_dummies(trainData.Education,prefix='education').education_Graduate
trainData.Self_Employed.value_counts()
self_emp_category = pd.get_dummies(trainData.Self_Employed,prefix='employed').employed_Yes
loan_status = pd.get_dummies(trainData.Loan_Status,prefix='status').status_Y
property_category = pd.get_dummies(trainData.Property_Area,prefix='property')
trainData.shape

trainNew = pd.concat([trainData,gender_cat,married_category,graduate_category,self_emp_category,loan_status,property_category],axis=1)
trainNew.head()
trainNew.columns
feature_columns = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','gender_Female','marriage_Yes','education_Graduate','employed_Yes','property_Rural','property_Semiurban','property_Urban']

X = trainNew[feature_columns]
y =  trainNew['status_Y']
y

from sklearn.cross_validation import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.01,random_state=42)
X_train.shape
X_test.shape

randForest = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
randForest.fit(X_train,y_train)
y_pred_class  = randForest.predict(X_test)
randForestScore = accuracy_score(y_test,y_pred_class)
get_ipython().magic(u'time print "Random forest accuraccy score",randForestScore')

#Import test data and do real test of our model
randForestNew = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
randForestNew.fit(X,y)
testData = pd.read_csv('H:/test.csv')
testData.shape
testData.head()
testData.isnull().sum()
testData.Gender.fillna(testData.Gender.max(),inplace =True)
testData.Married.fillna(testData.Married.max(),inplace=True)
testData.Credit_History.fillna(testData.Credit_History.max(),inplace=True)
testData.LoanAmount.fillna(testData.LoanAmount.mean(),inplace=True)
testData.Loan_Amount_Term.fillna(testData.Loan_Amount_Term.mean(),inplace=True)
testData.Self_Employed.fillna(testData.Self_Employed.max(),inplace=True)
testData.Dependents.fillna(0,inplace=True)

gender_cat = pd.get_dummies(testData.Gender,prefix='gender').gender_Female
married_category = pd.get_dummies(testData.Married,prefix='marriage').marriage_Yes
graduate_category = pd.get_dummies(testData.Education,prefix='education').education_Graduate
self_emp_category = pd.get_dummies(testData.Self_Employed,prefix='employed').employed_Yes
property_category = pd.get_dummies(testData.Property_Area,prefix='property')

testDataNew = pd.concat([testData,gender_cat,married_category,graduate_category,self_emp_category,property_category],axis=1)

X_testData = testDataNew[feature_columns]

X_testData.head()

y_test_pread_class = randForestNew.predict(X_testData)

randForestFormat = ["Y" if i == 1 else "N" for i in y_test_pread_class ]

pd.DataFrame({'Loan_ID':testData.Loan_ID,'Loan_Status':randForestFormat}).to_csv('radom_forest_submission.csv',index=False)

#Solve using logistic regression

from sklearn.linear_model import LogisticRegression

logReg = LogisticRegression()
logReg.fit(X_train,y_train)
logREg_predict =logReg.predict(X_test)
accuracy_score(y_test,logREg_predict)

logReg_y_prediction_class = logReg.predict(X_testData)

logRegPredictionFormat = ["Y" if i == 1 else "N" for i in logReg_y_prediction_class ]

#zip(logRegPredictionFormat,logReg_y_prediction_class)

pd.DataFrame({'Loan_ID':testData.Loan_ID,'Loan_Status':logRegPredictionFormat}).to_csv('logReg_submission.csv',index=False)

