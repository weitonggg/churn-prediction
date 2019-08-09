import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# random forest with no hyperparameter tuning
rf_base = RandomForestClassifier(random_state=1)
rf_base.fit(x_train, y_train)

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
'''
- Random Forest Classifier (no tuning),
 'F1: 0.6836960625945363, accuracy: 0.7780407004259347, 
 precision: 0.7115073833909056, recall: 0.6699164232392081')
- Random Forest Classifier with Randomized Search,
 'F1: 0.7209505785840007, accuracy: 0.8007572172266919, 
 precision: 0.7459361052384308, recall: 0.7063123058376223')
- Random Forest Classifier with Grid Search,
 'F1: 0.7186574575013546, accuracy: 0.8017037387600567, 
 precision: 0.7490061528626792, recall: 0.7022841609866927')
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