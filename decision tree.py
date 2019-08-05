import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# fit decision tree model
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

# prediction
dt_pred = dt.predict(x_test)

# f1 score
f1_score(y_test, dt_pred, average = 'macro') # 0.6378

# ROC curve & AUC
dt_pred_proba = dt.predict_proba(x_test)[:,1]
y_test_int = y_test.astype(int)
fpr,tpr,thresholds = roc_curve(np.array(y_test_int), dt_pred_proba)
plt.plot(fpr,tpr)
plt.show()
auc = roc_auc_score(np.array(y_test_int), dt_pred_proba) # 0.6391
