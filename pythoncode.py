import pandas as pd
import numpy as np

from tabulate import tabulate


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

DSLoanTrain = pd.read_csv("C:/work/DS/lendingclubloan/loan2.csv")
DSLoanTrain.head()

CleanData = DSLoanTrain.select_dtypes(include=[np.number]).interpolate().dropna()

yPredict = CleanData.loan_status
XClean = CleanData.drop(["loan_status"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(XClean, yPredict, random_state=42, test_size=.33)
clf = RandomForestRegressor(n_jobs=2, n_estimators=1000)
model = clf.fit(X_train, y_train)


headers = ["name", "score"]
values = sorted(zip(X_train.columns, model.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt="plain"))


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
#scaler.fit(X_train)

import pandas as pd
import numpy as np

DSLoanTrain2 = pd.read_csv("C:/work/DS/lendingclubloan/loan-clean-version1.csv")
class_mapping = {label:idx for idx, label in enumerate(np.unique(DSLoanTrain2['term']))}
DSLoanTrain2['term']=DSLoanTrain2['term'].map(class_mapping)

class_mapping = {label:idx for idx, label in enumerate(np.unique(DSLoanTrain2['grade']))}
DSLoanTrain2['grade']=DSLoanTrain2['grade'].map(class_mapping)

class_mapping = {label:idx for idx, label in enumerate(np.unique(DSLoanTrain2['emp_length']))}
DSLoanTrain2['emp_length']=DSLoanTrain2['emp_length'].map(class_mapping)

class_mapping = {label:idx for idx, label in enumerate(np.unique(DSLoanTrain2['home_ownership']))}
DSLoanTrain2['home_ownership']=DSLoanTrain2['home_ownership'].map(class_mapping)

class_mapping = {label:idx for idx, label in enumerate(np.unique(DSLoanTrain2['verification_status']))}
DSLoanTrain2['verification_status']=DSLoanTrain2['verification_status'].map(class_mapping)

class_mapping = {label:idx for idx, label in enumerate(np.unique(DSLoanTrain2['purpose']))}
DSLoanTrain2['purpose']=DSLoanTrain2['purpose'].map(class_mapping)

class_mapping = {label:idx for idx, label in enumerate(np.unique(DSLoanTrain2['addr_state']))}
DSLoanTrain2['addr_state']=DSLoanTrain2['addr_state'].map(class_mapping)

class_mapping = {label:idx for idx, label in enumerate(np.unique(DSLoanTrain2['loan_status']))}
DSLoanTrain2['loan_status']=DSLoanTrain2['loan_status'].map(class_mapping)

from sklearn.model_selection import train_test_split

DSLoanTrain2.head()

DSLoanTrain2 = DSLoanTrain2.select_dtypes(include=[np.number]).interpolate().dropna()
DSLoanTrain2 = DSLoanTrain2.drop(["total_pymnt"], axis=1)
DSLoanTrain2 = DSLoanTrain2.drop(["total_pymnt_inv"], axis=1)
DSLoanTrain2 = DSLoanTrain2.drop(["total_rec_int"], axis=1)


yPredict = DSLoanTrain2.loan_status
XClean = DSLoanTrain2.drop(["loan_status"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(XClean, yPredict, random_state=42, test_size=.30)

yPredict.head()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))



print(classification_report(y_test,predictions))
