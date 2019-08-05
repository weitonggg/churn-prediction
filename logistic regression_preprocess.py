import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

data = pd.read_csv('Telco-Customer-Churn.csv',  na_values = " ")

# data preprocessing
data['SeniorCitizen'] = data['SeniorCitizen'].astype(object)
data['TotalCharges'] = data['TotalCharges'].astype(float)
data['Churn'] = data['Churn'].replace({'Yes':'1', 'No':'0'})
data.TotalCharges = data.TotalCharges.fillna(data.MonthlyCharges)

data.gender = data.gender.replace({'Female':'0', 'Male':'1'})
data.Partner = data.Partner.replace({'Yes':'1', 'No':'0'})
data.Dependents = data.Dependents.replace({'Yes':'1', 'No':'0'})
data.PhoneService = data.PhoneService.replace({'Yes':'1', 'No':'0'})

## dummy columns for categorical features
multiple_phonelines_dummy = pd.get_dummies(data.MultipleLines).rename(columns = lambda x: 'multiplelines_' + str(x)).astype(object)
internet_service_dummy = pd.get_dummies(data.InternetService).rename(columns = lambda x: 'internetservice_' + str(x)).astype(object)
online_security_dummy = pd.get_dummies(data.OnlineSecurity).rename(columns = lambda x: 'onlinesecurity_' + str(x)).astype(object)
online_backup_dummy = pd.get_dummies(data.OnlineBackup).rename(columns = lambda x: 'onlinebackup_' + str(x)).astype(object)
device_protect_dummy = pd.get_dummies(data.DeviceProtection).rename(columns = lambda x: 'deviceprotect_' + str(x)).astype(object)
tech_support_dummy = pd.get_dummies(data.TechSupport).rename(columns = lambda x: 'techsupport_' + str(x)).astype(object)
streamtv_dummy = pd.get_dummies(data.StreamingTV).rename(columns = lambda x: 'streamtv_' + str(x)).astype(object)
streammovies_dummy = pd.get_dummies(data.StreamingMovies).rename(columns = lambda x: 'streammovies_' + str(x)).astype(object)
contract_dummy = pd.get_dummies(data.Contract).rename(columns = lambda x: 'contract_' + str(x)).astype(object)
paperless_bill_dummy = pd.get_dummies(data.PaperlessBilling).rename(columns = lambda x: 'paperlessbill_' + str(x)).astype(object)
pay_method_dummy = pd.get_dummies(data.PaymentMethod).rename(columns = lambda x: 'paymethod_' + str(x)).astype(object)

df = pd.concat([data, multiple_phonelines_dummy, internet_service_dummy,
                online_security_dummy, online_backup_dummy,
                device_protect_dummy, tech_support_dummy,
                streamtv_dummy, streammovies_dummy,
                contract_dummy, paperless_bill_dummy,
                pay_method_dummy],axis = 1)

df = df.drop(['customerID','MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], axis = 1)

## separate data into train-test
X= df.drop(['Churn'], axis=1)
y=df.Churn
x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state=0)

# fit logistic regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)
# predict 
lr_pred = lr.predict(x_test)

# evaluation metric: f1 score
f1_score(y_test, lr_pred, average = 'macro') # 0.7277

# evaluation metric: auc
lr_pred_proba = lr.predict_proba(x_test)[:,1]
y_test_int = y_test.astype(int)
fpr,tpr,thresholds = roc_curve(np.array(y_test_int), lr_pred_proba)
plt.plot(fpr,tpr)
plt.show()
auc = roc_auc_score(np.array(y_test_int), lr_pred_proba) # 0.8343
