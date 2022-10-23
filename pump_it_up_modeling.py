# -*- coding: utf-8 -*-
"""Pump It Up Modeling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1_jFfCJORaO1xq6FtlbNhRLqDXesK_qso

# Featue Engineering
"""

from google.colab import drive
drive.mount('/content/drive/')

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/drive/My Drive/Pump/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import scipy as sp
import matplotlib as mpl
import seaborn as sns

"""#### Importing Processed Dataset"""

train = pd.read_csv('training_data.csv')
test = pd.read_csv('test_data.csv')

training = train.drop('status_group', axis=1)
training.head()

training = training.drop('Unnamed: 0', axis=1)
test_for_submission = test.drop('Unnamed: 0', axis=1)

training.head()

"""#### Categorical Data Transformation using Custom Feature"""

def transform_feature(df, column_name):
    ''' Categorical data transformation based on unique values'''
    
    unique_values = set(df[column_name].tolist())
    transformer_dict = {}
    for index, value in enumerate(unique_values):
        transformer_dict[value] = index
    df[column_name] = df[column_name].apply(lambda y: transformer_dict[y])
    return df

numerical_columns = ['days_since_recorded', 'population','gps_height','amount_tsh','longitude','latitude'] 
columns_to_transform = [col for col in training.columns if col not in numerical_columns]
for column in columns_to_transform: 
    training = transform_feature(training, column)
    test_for_submission = transform_feature(test_for_submission, column)

"""#### Numerical Data Transformation using Custom Feature"""

#Reference - https://stackoverflow.com/questions/26414913/normalize-columns-of-a-dataframe

def normalize(df):
    '''Normalizes Column of a dataframe'''

    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

training = normalize(training)
test_for_submission = normalize(test_for_submission)

"""# Visulaiztion After Transformation of Features"""

from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
import pandas as pd

X = training
y = train['status_group']

tsne = TSNE(n_components=2,perplexity=10, verbose=1, random_state=123)
z = tsne.fit_transform(X)
df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 3),
                data=df).set(title="Pump data T-SNE projection")

labels = list(train['status_group'])
df_temp=training
df_temp['status_group'] = labels
df_temp.head()

sns.pairplot(df_temp, hue="status_group")

"""**The data is very non-linear and hard to find any pattern using hindsight. One thing is clear Linear Models won't work on such dataset.**

**We must go for Ensemble models for efficient Solution such as RF and GBDT.**

# Model Selection

###  Converting the Training dataframe into a matrix
"""

# Converting the Training dataframe into a matrix
X = training.to_numpy()

"""### Converting Class labels using Label Encoder"""

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(train.status_group)
train['categorical_label'] = le.transform(train.status_group)

list(le.classes_)

y = train["categorical_label"].tolist()

X.shape

len(y)

"""### Splitting the dataset into Test and Train"""

import sklearn.model_selection 
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.3, random_state = 32)

"""## Different Model Experimentation

### Logistic Regression
"""

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform

LR_model = LogisticRegression()

# define evaluation
cv_LR = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# define search space
space = dict()
space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
space['C'] = loguniform(1e-5, 100)

# define search
search_LR = RandomizedSearchCV(LR_model, space, n_iter=10, scoring='f1_weighted', n_jobs=-1, cv=cv_LR, random_state=1,verbose=10)

# execute search
result_LR = search_LR.fit(X_train, y_train)

best_LR_model=result_LR.best_estimator_

"""##### Score and Confusion Matrix"""

best_LR_model.score(X_train,y_train)

best_LR_model.score(X_test,y_test)

# Plotting Confusion Matrix for Train Data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmd =  ConfusionMatrixDisplay.from_estimator(best_LR_model,X_train,y_train, display_labels=['functional','needs repair','non functional'],cmap='Blues')
plt.grid(False)
plt.title('Confusion Matrix for Train Data')

"""### XGBoost"""

#Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost
classifier = xgboost.XGBClassifier()

params = {
 'learning_rate' : [0.05,0.10,0.15,0.20,0.25,0.30],
 'max_depth' : [ 3, 4, 5, 6, 8, 10, 12, 15],
 'min_child_weight' : [ 1, 3, 5, 7 ],
 'gamma': [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 'colsample_bytree' : [ 0.3, 0.4, 0.5 , 0.7 ]
}

xg_rs_ba_model=RandomizedSearchCV(classifier,param_distributions=params,n_iter=50,scoring='balanced_accuracy',n_jobs=-1,cv=5,verbose=3)

xg_rs_ba_model.fit(X_train,y_train)

xg_rs_ba_model.best_estimator_

"""##### Balanced Accurcay Score"""

xg_rs_ba_model.score(X_train,y_train)

xg_rs_ba_model.score(X_test,y_test)

xg_rs_f1m_model=RandomizedSearchCV(classifier,param_distributions=params,n_iter=50,scoring='f1_macro',n_jobs=-1,cv=5,verbose=3)

xg_rs_f1m_model.fit(X_train,y_train)

xg_rs_f1m_model.best_estimator_

"""##### f1_macro Score"""

xg_rs_f1m_model.score(X_train,y_train)

xg_rs_f1m_model.score(X_test,y_test)

xg_rs_model=RandomizedSearchCV(classifier,param_distributions=params,n_iter=50,scoring='f1_weighted',n_jobs=-1,cv=5,verbose=3)

#model fitting
xg_rs_model.fit(X_train,y_train)

#parameters selected
best_xg_model = xg_rs_model.best_estimator_

"""##### f1_weighted score"""

best_xg_model.score(X_train,y_train)

best_xg_model.score(X_test,y_test)

# Plotting Confusion Matrix for Train Data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmd =  ConfusionMatrixDisplay.from_estimator(best_xg_model,X_train,y_train, display_labels=['functional','needs repair','non functional'],cmap='Blues')
plt.grid(False)
plt.title('Confusion Matrix for Train Data')

# Plotting Confusion Matrix for Test Data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmd =  ConfusionMatrixDisplay.from_estimator(best_xg_model,X_test,y_test, display_labels=['functional','needs repair','non functional'],cmap='Blues')
plt.grid(False)
plt.title('Confusion Matrix for Test Data')

"""### Decision Tree"""

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist1 = {"max_depth": range(1,3000),
              "max_features": randint(1, 50),
              "min_samples_leaf": randint(1, 20),
              "criterion": ["gini", "entropy"]}

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv1 = RandomizedSearchCV(tree, param_dist1, cv=100)

tree_cv1.fit(X_train,y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv1.best_params_))
print("Best score is {}".format(tree_cv1.best_score_))

tree_best=tree_cv1.best_estimator_

"""##### Score"""

tree_best.score(X_train,y_train)

tree_best.score(X_test,y_test)

"""##### Confusion Matrix"""

from sklearn.metrics import confusion_matrix

# Constructing the Confusion Matrix for Train Data
cm = confusion_matrix(y_train, tree_best.predict(X_train))
np.set_printoptions(precision=2)
print('Confusion matrix for Train Data')
print(cm)

# Plotting Confusion Matrix for Train Data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmd =  ConfusionMatrixDisplay.from_estimator(tree_best,X_train,y_train, display_labels=['functional','needs repair','non functional'],cmap='Blues')
plt.grid(False)
plt.title('Confusion Matrix for Train Data')

from sklearn.metrics import confusion_matrix

# Constructing the Confusion Matrix for Test Data
cm = confusion_matrix(y_test, tree_best.predict(X_test))
np.set_printoptions(precision=2)
print('Confusion matrix for Test Data')
print(cm)

# Plotting Confusion Matrix for Test Data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmd =  ConfusionMatrixDisplay.from_estimator(tree_best,X_test,y_test, display_labels=['functional','needs repair','non functional'],cmap='Blues')
plt.grid(False)
plt.title('Confusion Matrix for Test Data')

"""### Random Forest Classifier"""

# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

"""#### RandomGridSearchCV to search for best hyperparameters : Scoring-Accuracy"""

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=10, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)

rf_random.best_params_

"""Above are the best parameters obtained for our data from training 300 models of various combinations of parameters using Random Search cross validation

##### Fitting Using Best RF Model Parameters
"""

rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=1600, 
                                              min_samples_split=5,
                                              criterion='gini',
                                              min_samples_leaf = 1, 
                                              max_features='auto',
                                              oob_score=True,
                                              max_depth = 90,
                                              random_state=42,
                                              bootstrap=True,
                                              n_jobs=-1)

rfc.fit(X_train, y_train)

"""##### Feature Importance"""

print(rfc.feature_importances_)

X_labels_features = list(training.columns.values)

importance = rfc.feature_importances_
importance = pd.DataFrame(importance, index=training.columns, 
                          columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_ 
                            for tree in rfc.estimators_], axis=0)

x = X_labels_features
y = importance.iloc[:, 0]
yerr = importance.iloc[:, 1]

ax = plt.bar(x, y, yerr=yerr ,align="center")
plt.xlabel('Features', fontsize = 12)
plt.ylabel('Importance', fontsize = 12)

plt.xticks(rotation = 90) 
plt.figure(figsize=(50,50)) 
plt.show()

"""Most of the features are equally important. It is seen that the new feature(days_since_recorded) added ahs been very useful.

##### Accuracy Score
"""

#Train Accuracy

print('Random Forest Classifier Train Accuracy Score :', np.round(100 * rfc.score(X_train, y_train),2))

#Test Accuracy

print('Random Forest Classifier Accuracy Test Score :', np.round(100 * rfc.score(X_test, y_test),2))

"""##### AUC-ROC Plot"""

#https://medium.com/swlh/how-to-create-an-auc-roc-plot-for-a-multiclass-model

from yellowbrick.classifier import ROCAUC

def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder={0: 'functional', 
                                        1: 'needs repair', 
                                        2: 'non functional'})
                                        
    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()
    
    return visualizer

plot_ROC_curve(rfc, X_train, y_train, X_test, y_test)

"""##### Classification Report"""

from sklearn.metrics import classification_report

y_pred = rfc.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

"""Most of the error is seen for class 1(needs repair). This is due to class imbalance and lack of appropriate data for the class in training set.

##### Confusion Matrix
"""

from sklearn.metrics import confusion_matrix

# Constructing the Confusion Matrix for Train Data
cm = confusion_matrix(y_train, rfc.predict(X_train))
np.set_printoptions(precision=2)
print('Confusion matrix for Train Data')
print(cm)

# Plotting Confusion Matrix for Train Data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmd =  ConfusionMatrixDisplay.from_estimator(rfc,X_train,y_train, display_labels=['functional','needs repair','non functional'],cmap='Blues')
plt.grid(False)
plt.title('Confusion Matrix for Train Data')

from sklearn.metrics import confusion_matrix

# Constructing the Confusion Matrix for Test Data
cm = confusion_matrix(y_test, rfc.predict(X_test))
np.set_printoptions(precision=2)
print('Confusion matrix for Test Data')
print(cm)

# Plotting Confusion Matrix for Test Data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmd =  ConfusionMatrixDisplay.from_estimator(rfc,X_test,y_test, display_labels=['functional','needs repair','non functional'],cmap='Blues')
plt.grid(False)
plt.title('Confusion Matrix for Test Data')

"""##### Visualizing Any One Tree in the Forest"""

important_feature_names = training.columns.to_list()

# Use sklearn to export the tree 
from sklearn.tree import export_graphviz

# Write the decision tree as a dot file
visual_tree = rfc.estimators_[12]
export_graphviz(visual_tree, out_file = 'best_tree.dot', feature_names = important_feature_names, 
                precision = 2, filled = True, rounded = True, max_depth = None)

# Use pydot for converting to an image file
import pydot

# Import the dot file to a graph and then convert to a png
(graph, ) = pydot.graph_from_dot_file('best_tree.dot')
graph.write_png('best_tree.png')

"""#### RandomGridSearchCV to search for best hyperparameters : Scoring-f1_score"""

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf_f1 = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_f1_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50,scoring='f1_weighted', cv = 3, verbose=10, random_state=42, n_jobs = -1)

rf_f1_random.fit(X_train, y_train)

rf_f1_random.best_params_

"""##### Fitting Using Best RF Model Parameters"""

rfc_f1 = sklearn.ensemble.RandomForestClassifier(n_estimators=200, 
                                              min_samples_split=5,
                                              min_samples_leaf=2,
                                              criterion='gini', 
                                              max_features='auto',
                                              oob_score=False,
                                              max_depth = 90,
                                              random_state=42,
                                              bootstrap=False,
                                              n_jobs=-1)

rfc_f1.fit(X_train, y_train)

"""##### Feature Importance"""

print(rfc_f1.feature_importances_)

X_labels_features = list(training.columns.values)

importance = rfc_f1.feature_importances_
importance = pd.DataFrame(importance, index=training.columns, 
                          columns=["Importance"])

importance["Std"] = np.std([tree.feature_importances_ 
                            for tree in rfc_f1.estimators_], axis=0)

x = X_labels_features
y = importance.iloc[:, 0]
yerr = importance.iloc[:, 1]

ax = plt.bar(x, y, yerr=yerr ,align="center")
plt.xlabel('Features', fontsize = 12)
plt.ylabel('Importance', fontsize = 12)

plt.xticks(rotation = 90) 
plt.figure(figsize=(50,50)) 
plt.show()

"""Most of the features are equally important. It is seen that the new feature(days_since_recorded) added has been very useful.

##### f1_weighted Score
"""

#Train Accuracy

print('Random Forest Classifier Train  Score :', np.round(100 * rfc_f1.score(X_train, y_train),2))

#Test Accuracy

print('Random Forest Classifier  Test Score :', np.round(100 * rfc_f1.score(X_test, y_test),2))

"""##### AUC-ROC Plot"""

#https://medium.com/swlh/how-to-create-an-auc-roc-plot-for-a-multiclass-model

from yellowbrick.classifier import ROCAUC

def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder={0: 'functional', 
                                        1: 'needs repair', 
                                        2: 'non functional'})
                                        
    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()
    
    return visualizer

plot_ROC_curve(rfc_f1, X_train, y_train, X_test, y_test)

"""##### Classification Report"""

from sklearn.metrics import classification_report

y_pred = rfc_f1.predict(X_test)

report = classification_report(y_test, y_pred)
print(report)

"""Most of the error is seen for class 1(needs repair). This is due to class imbalance and lack of appropriate data for the class in training set.

##### Confusion Matrix
"""

from sklearn.metrics import confusion_matrix

# Constructing the Confusion Matrix for Train Data
cm = confusion_matrix(y_train, rfc_f1.predict(X_train))
np.set_printoptions(precision=2)
print('Confusion matrix for Train Data')
print(cm)

# Plotting Confusion Matrix for Train Data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmd =  ConfusionMatrixDisplay.from_estimator(rfc_f1,X_train,y_train, display_labels=['functional','needs repair','non functional'],cmap='Blues')
plt.grid(False)
plt.title('Confusion Matrix for Train Data')

from sklearn.metrics import confusion_matrix

# Constructing the Confusion Matrix for Test Data
cm = confusion_matrix(y_test, rfc_f1.predict(X_test))
np.set_printoptions(precision=2)
print('Confusion matrix for Test Data')
print(cm)

# Plotting Confusion Matrix for Test Data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmd =  ConfusionMatrixDisplay.from_estimator(rfc_f1,X_test,y_test, display_labels=['functional','needs repair','non functional'],cmap='Blues')
plt.grid(False)
plt.title('Confusion Matrix for Test Data')

"""### Model Selection Reasoning - **RANDOM FOREST**

1. We can observe that a single DT is less accurate and reliable than Random Forest.

2. The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability/robustness over a single estimator.

3. More trees improves the performance and make predictions more stable but also slow down the computation speed.

4. The range of predictions a Random Forest can make is bound by the highest and lowest labels in the training data. As the features we have are standard in nature, we won't face such extreme data point problem.

5. Most importantly due to non-linear nature of dataset,we will use RF which is generally used for classifying non-linearly separable data.

### Model Summary
"""

from prettytable import PrettyTable
  
# Specify the Column Names while initializing the Table
myTable = PrettyTable(["Model", "Scoring", "Train Score", "Test Score"])
  
# Add rows
myTable.add_row(["Logistic Regression", "f1_weighted", "0.627", "0.625"])
myTable.add_row(["XGBoost", "Balanced Accuracy", "0.937", "0.662"])
myTable.add_row(["XGBoost", "f1_weighted", "0.964", "0.81"])
myTable.add_row(["XGBoost", "f1_macro", "0.937", "0.68"])
myTable.add_row(["Decision Tree", "Accuracy", "0.81", "0.76"])
myTable.add_row(["Random Forest", "Accuracy", "0.966", "0.81"])
myTable.add_row(["Random Forest", "f1_weighted", "0.975", "0.812"])

  
print(myTable)

"""**1.** **Changing some parameters while using f1 scoring, gained me 5% more true classification for class 1 which is critical.**

**2.** **I was abel to reduce no, of estimators from 1600 to 200 which will help the model to perform faster evaluation.**

### Saving our Best Models
"""

import pickle

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(rfc_f1, open(filename, 'wb'))

# save a different model to disk
filename = 'finalized_model_xgboost.sav'
pickle.dump(best_xg_model, open(filename, 'wb'))

# save a different model to disk
filename = 'finalized_model_dt.sav'
pickle.dump(tree_best, open(filename, 'wb'))