# EDA and preprocessing imports
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
# metrics and scoring
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
# models
import xgboost as xgb
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tpot import TPOTClassifier

# view and change current working directory
path = os.getcwd()
print(path)
os.chdir('/Users/david/Desktop/python_datasets')
print(os.getcwd())

# import dataset
diabetes_data = pd.read_csv("diabetes_data.csv")
# get python to show all column, remove truncation of columns
pd.options.display.max_columns = 10

# EDA

print(diabetes_data.head())
# look for missing values
print(diabetes_data.isnull().sum())
# no missing values

# Print datatypes and shape of dataset
print(diabetes_data.dtypes)
print(diabetes_data.shape)
# get counts of outcome variable
print(diabetes_data['Outcome'].value_counts())

# look at some descriptive statistics of the data
# notes on glucose: < 140 normal   140<x<199 is pre-diabetic and 200+ is diabetic
diabetes_data.describe()
# variance of variables
print(diabetes_data.var())

# separate dependent and independent variables
X = diabetes_data.drop(['Outcome'], axis=1).values
y = diabetes_data['Outcome'].values

# create a count plot of Outcome data
Outcome_countplot = sns.countplot(x='Outcome', data=diabetes_data)

# scatter matrix must be a dataframe. had to convert X from an  array to X_df which is a DataFrame
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age']
X_df = pd.DataFrame(X, columns = column_names)
print(X_df)
pd.plotting.scatter_matrix(X_df, c=y, figsize=[9, 9], s=150, marker='D')
plt.show()

# will not use stratify=y because data is not imbalanced
# using stratify=y decreased testing accuracy by about 4%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# LogisticRegression gave a convergence warning. had to change the max_iter to a higher value
logreg = LogisticRegression(max_iter=1000, random_state=1)
logreg.fit(X_train, y_train)
logreg_test_accuracy = logreg.score(X_test, y_test)
print('Logistic Regression Accuracy is:', logreg_test_accuracy)
y_pred_logreg = logreg.predict(X_test)
y_pred_logreg_prob = logreg.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_logreg_prob)
lr_accuracy_score = accuracy_score(y_test, y_pred_logreg)
# log reg accuracy score is 0.7792
print('ROC AUC Score for Logistic Regression is:', auc_score)
# AUC score is 0.8417


# use AUC with cross validation, compare cv=5 with cv=10
# mean AUC score with CV=5 is 0.833
logreg_cv5_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
print('Mean AUC score with 5 fold cross validation is:', np.mean(logreg_cv5_scores))
# mean AUC score with CV=10 is 0.829
logreg_cv10_scores = cross_val_score(logreg, X, y, cv=10, scoring='roc_auc')
print('Mean AUC score with 10 fold cross validation is:', np.mean(logreg_cv10_scores))

# Use K nearest neighbors
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(X_train, y_train)
y_pred_knn_5 = knn_5.predict(X_test)
print('KNN=5 score is:', knn_5.score(X_test, y_test))

# plot to show how KNN Classifier accuracy changes with each neighbor
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1, 100)
for n_neighbors in neighbors_settings:
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    training_accuracy.append(knn.score(X_train, y_train))
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label='training accuracy')
plt.plot(neighbors_settings, test_accuracy, label='test accuracy')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# knn with neighbors=5 is 0.7338  neighbors=8 is 0.7662    neighbors=16 is 0.7857

knn_16 = KNeighborsClassifier(n_neighbors=16)
knn_16.fit(X_train, y_train)
y_pred_knn_16 = knn_5.predict(X_test)
print('KNN=16 score is:', knn_16.score(X_test, y_test))

# Use GridSearchCV to combine KNN and CV
param_grid = {'n_neighbors': np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X_train, y_train)
print('KNN Gridsearched best parameters are:', knn_cv.best_params_)
print('KNN Gridsearched best score is:',knn_cv.best_score_)
# KNN with cv n_neighbors=25, best score= 0.7346

# Ridge (penalty='l2') is default for logistic regression, compare with Lasso (penalty='l1')
logreg_l1 = LogisticRegression(C=1, penalty='l1', solver='liblinear')
logreg_l1.fit(X_train, y_train)
logreg_l1_test_accuracy = logreg_l1.score(X_test, y_test)
print('L1 Logistic Regression Test Accuracy:', logreg_l1_test_accuracy)

logreg_l2 = LogisticRegression(C=1, penalty='l2', solver='liblinear')
logreg_l2.fit(X_train, y_train)
logreg_l2_test_accuracy = logreg_l2.score(X_test, y_test)
print('L2 Logistic Regression Test Accuracy:', logreg_l2_test_accuracy)
# all are very close but l2 liblinear has the highest accuracy


# Combine Scaler with KNN using a pipeline
steps = [('scaler', StandardScaler()),
         ('knn', KNeighborsClassifier(n_neighbors=5))]
pipeline = Pipeline(steps)
knn_scaled = pipeline.fit(X_train, y_train)
knn_y_pred = pipeline.predict(X_test)
knn_scaled_accuracy_score = accuracy_score(y_test, knn_y_pred)
print('KNN with Standard Scaler Accuracy Score:', knn_scaled_accuracy_score)
# accuracy score 0.7987


# fit with DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
dt.fit(X_train, y_train)
dt_y_pred = dt.predict(X_test)
dt_accuracy_score = accuracy_score(y_test, dt_y_pred)
print('Decision Tree Classifier Accuracy Score:', dt_accuracy_score)
# DT accuracy score=0.7987


# Voting Classifier
lr = LogisticRegression(max_iter=1000, random_state=1)
knn_5 = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(max_depth=2, random_state=1)

classifiers = [('LogisticRegression()', lr),
               ('K Nearest Neighbors()', knn_5),
               ('Classification Tree', dt)]
vc = VotingClassifier(estimators=classifiers)
vc.fit(X_train, y_train)
vc_y_pred = vc.predict(X_test)
vc_accuracy_score = accuracy_score(y_test, vc_y_pred)
print('Voting Classifier with Logistic Regression, KNN and Decision Trees Accuracy Score:', vc_accuracy_score)
# accuracy score is 0.7922, using KNN=16 was not beneficial to the accuracy


# Bagging Classifier
dt = DecisionTreeClassifier(max_depth=2, random_state=1)
bc = BaggingClassifier(base_estimator=dt, n_estimators=300, n_jobs=-1)
bc.fit(X_train, y_train)
bc_y_pred = bc.predict(X_test)
bc_accuracy_score = accuracy_score(y_test, bc_y_pred)
print('Bagging (Bootstrap Aggregation) Accuracy Score:', bc_accuracy_score)

oob_bc = BaggingClassifier(base_estimator=dt, n_estimators=300,
                           oob_score=True, n_jobs=-1)
oob_bc.fit(X_train, y_train)
oob_bc_y_pred = oob_bc.predict(X_test)
oob_bc_accuracy_score = oob_bc.oob_score_
print('Out of Bag Classifier Accuracy Score:', oob_bc_accuracy_score)
# oob_score_ less than only decision tree

# use Random Forest Regressor
rf = RandomForestClassifier(n_estimators=400,
                            random_state=1)
rf.fit(X_train, y_train)
rf_y_pred = rf.predict(X_test)
rf_accuracy_score = rf.score(X_test, y_test)
print('Random Forest Classifier with 400 estimators accuracy score:', rf_accuracy_score)


# combine randomized search cv with random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestClassifier()
rf_random_cv = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=5, verbose=2,
                                  random_state=1, n_jobs=-1)
rf_random_cv.fit(X_train, y_train)
rf_random_cv.best_params_
# best parameters are n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto',
# 'max_depth': 40, 'bootstrap': False

rf_best = RandomForestClassifier(n_estimators=200, min_samples_split=2, min_samples_leaf=1, max_features='auto',
                                 max_depth=40, bootstrap=False)
rf_best.fit(X_train, y_train)
rf_best.predict(X_test)
rf_best_score = rf_best.score(X_test, y_test)
print('Best hyper-parameters from RandomSearchCV with RandomForest accuracy score', rf_best_score)
# Score from best CV model is still less than the generic RF model


# XGBOOST
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=1)
xg_cl.fit(X_train, y_train)
xg_y_pred = xg_cl.predict(X_test)
# xg_accuracy = accuracy_score(y_test, xg_y_pred)
# the above accuracy score doesn't work
xg_accuracy = float(np.sum(xg_y_pred == y_test)) / y_test.shape[0]
print('Accuracy of XGBOOST is:', xg_accuracy)


# use cross-validation with XGBoost
diabetes_dmatrix = xgb.DMatrix(data=X, label=y)
params = {'objective': 'binary:logistic', 'max_depth': 4}
cv_xg_results = xgb.cv(dtrain=diabetes_dmatrix, params=params,
                       nfold=4, num_boost_round=10, metrics='error',
                       as_pandas=True)
print('Accuracy of cross validation with XGBOOST is:', (1 - cv_xg_results["test-error-mean"]).iloc[-1])
# accuracy of cv_xb = 0.7552


# Use TPOT Classifier
tpot = TPOTClassifier(generations=3, population_size=5, verbosity=2, offspring_size=10,
                      scoring='accuracy', cv=5)
tpot.fit(X_train, y_train)
print('Accuracy score for TPOT Classifier is:', tpot.score(X_test, y_test))

# RF was the best model. create confusion matrix plot
con_matrix = confusion_matrix(y_test, rf_y_pred)
con_matrix_plot = sns.heatmap(con_matrix, annot=True, cmap='Blues')
con_matrix_plot.set_title('Confusion Matrix of Best Performing Model')
con_matrix_plot.set_xlabel('Predicted Values')
con_matrix_plot.set_ylabel('Actual Values');
con_matrix_plot.xaxis.set_ticklabels(['False', 'True'])
con_matrix_plot.yaxis.set_ticklabels(['False', 'True'])
plt.show()