import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filename):
    data = pd.read_csv(filename, na_values = ' ')
    return data
    
def data_preprocess(df):
    df['SeniorCitizen'] = df['SeniorCitizen'].astype(object)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df.TotalCharges = df.TotalCharges.fillna(df.MonthlyCharges)
    df.Churn = df.Churn.replace({'Yes':'1', 'No':'0'})
    df.gender = df.gender.replace({'Female':'0', 'Male':'1'})
    df.Partner = df.Partner.replace({'Yes':'1', 'No':'0'})
    df.Dependents = df.Dependents.replace({'Yes':'1', 'No':'0'})
    df.PhoneService = df.PhoneService.replace({'Yes':'1', 'No':'0'})
    
    multiple_phonelines_dummy = pd.get_dummies(df.MultipleLines).rename(columns = lambda x: 'multiplelines_' + str(x)).astype(object)
    internet_service_dummy = pd.get_dummies(df.InternetService).rename(columns = lambda x: 'internetservice_' + str(x)).astype(object)
    tech_support_dummy = pd.get_dummies(df.TechSupport).rename(columns = lambda x: 'techsupport_' + str(x)).astype(object)
    streamtv_dummy = pd.get_dummies(df.StreamingTV).rename(columns = lambda x: 'streamtv_' + str(x)).astype(object)
    streammovies_dummy = pd.get_dummies(df.StreamingMovies).rename(columns = lambda x: 'streammovies_' + str(x)).astype(object)
    contract_dummy = pd.get_dummies(df.Contract).rename(columns = lambda x: 'contract_' + str(x)).astype(object)
    paperless_bill_dummy = pd.get_dummies(df.PaperlessBilling).rename(columns = lambda x: 'paperlessbill_' + str(x)).astype(object)
    pay_method_dummy = pd.get_dummies(df.PaymentMethod).rename(columns = lambda x: 'paymethod_' + str(x)).astype(object)
    
    df = pd.concat([df, multiple_phonelines_dummy, internet_service_dummy,
                tech_support_dummy, streamtv_dummy, streammovies_dummy,
                contract_dummy, paperless_bill_dummy,
                pay_method_dummy],axis = 1)
    
    df = df.drop(['customerID','MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'], axis = 1)
    
    return df

def split_data(df, target, test_size):
    X = df.drop([target], axis=1)
    y = df[target]
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = test_size, random_state=0)
    return x_train, x_test, y_train, y_test


filename = 'Telco-Customer-Churn.csv'
data = load_data(filename)
data = data_preprocess(data)
x_train, x_test, y_train, y_test = split_data(data, 'Churn', 0.3)
