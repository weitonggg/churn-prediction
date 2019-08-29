import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# random forest with no hyperparameter tuning
rf_base = RandomForestClassifier(random_state=1)
rf_base.fit(x_train, y_train)

# Feature importance
feature_importances = pd.DataFrame(rf_base.feature_importances_,
                                   index = x_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)

# random forest with randomized search CV
n_estimators = np.arange(200,2200,200)
max_features = ['auto', 'sqrt']
max_depth = np.arange(10,110,20)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

rf_random = RandomizedSearchCV(RandomForestClassifier(), random_grid, n_iter = 50, cv = 3, 
                               random_state=1)
rf_random.fit(x_train, y_train)
rf_random.best_params_

# random forest with grid search CV
param_grid = {'bootstrap': [True],
 'max_features': [3,4,5],
 'min_samples_leaf': [2,4,6],
 'min_samples_split': [2,4,6],
 'n_estimators': [1000, 1200, 1400]}

rf_grid = GridSearchCV(RandomForestClassifier(), param_grid, cv = 3)
rf_grid.fit(x_train, y_train)
rf_grid.best_params_

# Adaboost
adb = AdaBoostClassifier(n_estimators=1000, learning_rate=0.1)
adb.fit(x_train, y_train)

# models evaluation
def evaluate_fit(model, x_test, y_test):
    preds = model.predict(x_test)
    f1 = f1_score(y_test, preds, average = 'macro')
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average = 'macro')
    recall = recall_score(y_test, preds, average = 'macro')
    
    return model.__class__.__name__, \
         'F1: {}, accuracy: {}, precision: {}, recall: {}' \
         .format(f1, accuracy, precision, recall)
         
evaluate_fit(rf_base, x_test, y_test)
evaluate_fit(rf_random.best_estimator_, x_test, y_test)
evaluate_fit(rf_grid.best_estimator_, x_test, y_test)
evaluate_fit(adb, x_test, y_test)

'''
- Random Forest Classifier (no tuning),
 'F1: 0.6836960625945363, accuracy: 0.7780407004259347, 
 precision: 0.7115073833909056, recall: 0.6699164232392081'
- Random Forest Classifier with Randomized Search,
 'F1: 0.7209505785840007, accuracy: 0.8007572172266919, 
 precision: 0.7459361052384308, recall: 0.7063123058376223'
- Random Forest Classifier with Grid Search,
 'F1: 0.7186574575013546, accuracy: 0.8017037387600567, 
 precision: 0.7490061528626792, recall: 0.7022841609866927'
- AdaBoost Classifier,
 'F1: 0.7332859218757697, accuracy: 0.8059630856601988, 
 precision: 0.7518386822270848, recall: 0.7209272267816571'
'''

# best params for RF using randomized search
'''
{'bootstrap': True,
 'max_features': 'sqrt',
 'min_samples_leaf': 4,
 'min_samples_split': 5,
 'n_estimators': 1200}
'''
# best params for RF using grid search
'''
{'bootstrap': True,
 'max_features':3,
 'min_samples_leaf': 6,
 'min_samples_split': 2,
 'n_estimators': 1000}
'''

'''
Feature Importance
                                     importance
TotalCharges                           0.195148
MonthlyCharges                         0.177853
tenure                                 0.171504
internetservice_Fiber optic            0.051548
contract_Month-to-month                0.037816
paymethod_Electronic check             0.033635
gender                                 0.027467
contract_Two year                      0.027233
Partner                                0.024208
SeniorCitizen                          0.023175
Dependents                             0.019505
techsupport_No                         0.018679
contract_One year                      0.018264
paperlessbill_No                       0.017569
paperlessbill_Yes                      0.016920
paymethod_Credit card (automatic)      0.013627
streammovies_No                        0.013078
paymethod_Bank transfer (automatic)    0.013066
multiplelines_Yes                      0.012952
streamtv_Yes                           0.012385
paymethod_Mailed check                 0.011914
streamtv_No                            0.011719
streammovies_Yes                       0.011518
techsupport_Yes                        0.010108
multiplelines_No                       0.010012
internetservice_DSL                    0.006731
techsupport_No internet service        0.004653
PhoneService                           0.003337
multiplelines_No phone service         0.002481
streamtv_No internet service           0.001171
internetservice_No                     0.000368
streammovies_No internet service       0.000358
'''
