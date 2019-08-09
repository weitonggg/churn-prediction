from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score


# Logistic regression model
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_pred = lr.predict(x_test)

f1_score(y_test, lr_pred, average = 'macro')
accuracy_score(y_test, lr_pred)
precision_score(y_test, lr_pred, average = 'macro')
recall_score(y_test, lr_pred, average = 'macro')

# Decision tree model
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
dt_pred = dt.predict(x_test)

f1_score(y_test, dt_pred, average = 'macro')
accuracy_score(y_test, dt_pred)
precision_score(y_test, dt_pred, average = 'macro')
recall_score(y_test, dt_pred, average = 'macro')

