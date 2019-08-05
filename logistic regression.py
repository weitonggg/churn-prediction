import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# fit logistic regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)

# predict 
lr_pred = lr.predict(x_test)

# f1 score
f1_score(y_test, lr_pred, average = 'macro') # 0.7277

# ROC curve & AUC
lr_pred_proba = lr.predict_proba(x_test)[:,1]
y_test_int = y_test.astype(int)
fpr,tpr,thresholds = roc_curve(np.array(y_test_int), lr_pred_proba)
plt.plot(fpr,tpr)
plt.show()
auc = roc_auc_score(np.array(y_test_int), lr_pred_proba) # 0.8343
