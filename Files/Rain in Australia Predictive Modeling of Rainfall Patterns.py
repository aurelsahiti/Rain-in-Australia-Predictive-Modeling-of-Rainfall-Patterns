#!/usr/bin/env python
# coding: utf-8

# # Load dataset

# ## Initialize dataset

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

# Need to manually upload dataset for now (Click on the folder icon on the right > right click > select "upload")
# TODO: Set up download automatically
df = pd.read_csv("./weatherAUS.csv")
df


# # Analysis

# ## Dataset Balance

# In[2]:


df['RainTomorrow'].value_counts().plot.pie(autopct='%.2f')


# # Preprocess

# In[3]:


# Remove records with no value for target attribute
df.dropna(subset=['RainTomorrow'], inplace=True)


# In[4]:


# Drop date column. May test with or without
df.drop("Date", axis="columns", inplace=True)


# ## Data imputation

# In[5]:


df.isna().sum()


# In[6]:


# Replace NaNs with the median for numerical features
df.fillna({"MinTemp": df["MinTemp"].median()}, inplace=True)
df.fillna({"MaxTemp": df["MaxTemp"].median()}, inplace=True)
df.fillna({"Rainfall": df["Rainfall"].median()}, inplace=True)
df.fillna({"Evaporation": df["Evaporation"].median()}, inplace=True)
df.fillna({"Sunshine": df["Sunshine"].median()}, inplace=True)
df.fillna({"WindGustSpeed": df["WindGustSpeed"].median()}, inplace=True)
df.fillna({"WindSpeed9am": df["WindSpeed9am"].median()}, inplace=True)
df.fillna({"WindSpeed3pm": df["WindSpeed3pm"].median()}, inplace=True)
df.fillna({"Humidity9am": df["Humidity9am"].median()}, inplace=True)
df.fillna({"Humidity3pm": df["Humidity3pm"].median()}, inplace=True)
df.fillna({"Pressure9am": df["Pressure9am"].median()}, inplace=True)
df.fillna({"Pressure3pm": df["Pressure3pm"].median()}, inplace=True)
df.fillna({"Cloud9am": df["Cloud9am"].median()}, inplace=True)
df.fillna({"Cloud3pm": df["Cloud3pm"].median()}, inplace=True)
df.fillna({"Temp9am": df["Temp9am"].median()}, inplace=True)
df.fillna({"Temp3pm": df["Temp3pm"].median()}, inplace=True)


# Replace NaNs with mode for categorical features
df.fillna({"WindGustDir": df["WindGustDir"].mode()[0]}, inplace=True)
df.fillna({"WindDir9am": df["WindDir9am"].mode()[0]}, inplace=True)
df.fillna({"WindDir3pm": df["WindDir3pm"].mode()[0]}, inplace=True)
df.fillna({"RainToday": df["RainToday"].mode()[0]}, inplace=True)

df.isna().sum()


# ## Label enconding

# In[7]:


from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
ohe_cols = enc.fit_transform(df[["Location", "WindGustDir", "WindDir9am", "WindDir3pm", 'RainToday']])
ohe_cols


# In[8]:


# Concat one-hot encoded columns
df = pd.concat([df, ohe_cols], axis=1)

# Drop redundant columns
df = df.drop(["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday"], axis=1)

df


# In[9]:


df['RainTomorrow'] = df['RainTomorrow'].replace(to_replace='No', value=0)
df['RainTomorrow'] = df['RainTomorrow'].replace(to_replace='Yes', value=1)


# ## Train test split

# In[10]:


y = df['RainTomorrow']
X = df.drop(['RainTomorrow'], axis=1)


# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.2, random_state=1, stratify=y)


# ## SMOTE

# In[12]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
y_train_res.value_counts().plot.pie(autopct='%.2f')


# ## Scale

# In[13]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_res)

# This should be just transform and only on the original X_test (not resampled)
X_test_scaled = scaler.transform(X_test)


# # Training and Predictions
# 
# 

# In[14]:


import multiprocessing

# Used for n_jobs in the GridSearchCV. Manually change n_cpus as needed.
n_cpus = multiprocessing.cpu_count()


# ## SVC

# ### Hyperparameter Tuning

# In[16]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

execute_cell = True

if execute_cell:
  # For sake of speed, opting not the include linear kernel. Takes hours...
  parameters = {'C': [0.5, 1, 5, 10]}
  svc = SVC(kernel='rbf', random_state=0)

  clf = GridSearchCV(svc, parameters, n_jobs=n_cpus)
  clf.fit(X_train_scaled, y_train_res)

  results_df = pd.DataFrame(clf.cv_results_)

  results_df = results_df.sort_values(by=["rank_test_score"])
  results_df = results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("name")
  display(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]])


# ### Train

# In[19]:


from sklearn.svm import SVC

svc = SVC(C=5, kernel='rbf', probability=True, random_state=0)

svc.fit(X_train_scaled, y_train_res)



# ### Evaluate

# In[ ]:


svc_y_pred_prob = svc.predict_proba(X_test_scaled)[:,1]
svc_y_pred = svc.predict(X_test_scaled)


# #### Accuracy and Classification Report

# In[18]:


from sklearn.metrics import accuracy_score, classification_report

print('Accuracy score:', accuracy_score(y_test, svc_y_pred))

print('Classifcation report:')
print(classification_report(y_test, svc_y_pred, digits=4))


# #### Confusion Matrix

# In[19]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmat = confusion_matrix(y_test, svc_y_pred)

cmatDisplay = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=svc.classes_)

cmatDisplay.plot()
plt.show()


# #### ROC

# In[20]:


from sklearn.metrics import roc_curve, roc_auc_score

### May not be accurate as the probability model is created using cross validation.

fpr, tpr, thresholds = roc_curve(y_test, svc_y_pred_prob, pos_label=1)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

print('ROC_AUC:', roc_auc_score(y_test, svc_y_pred_prob))


# ## Random Forest

# ### Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

execute_cell = True

if execute_cell:
  parameters = {'criterion': ['gini', 'entropy'], 'n_estimators': [100, 200, 300]}
  random_forest = RandomForestClassifier(random_state=0)

  clf = GridSearchCV(random_forest, parameters, n_jobs=n_cpus)

  clf.fit(X_train_scaled, y_train_res)

  results_df = pd.DataFrame(clf.cv_results_)

  results_df = results_df.sort_values(by=["rank_test_score"])
  results_df = results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("name")
  display(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]])


# ### Training

# In[1]:


from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=0)
random_forest.fit(X_train_scaled, y_train_res)


# ### Evaluation

# In[23]:


random_forest_y_pred_prob = random_forest.predict_proba(X_test_scaled)[:,1]
random_forest_y_pred = random_forest.predict(X_test_scaled)


# #### Accuracy and Classification Report

# In[24]:


from sklearn.metrics import accuracy_score, classification_report

print('Accuracy score:', accuracy_score(y_test, random_forest_y_pred))

print('Classifcation report:')
print(classification_report(y_test, random_forest_y_pred, digits=4))


# #### Confusion matrix

# In[25]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmat = confusion_matrix(y_test, random_forest_y_pred)

cmatDisplay = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=random_forest.classes_)

cmatDisplay.plot()
plt.show()


# #### ROC

# In[26]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, random_forest_y_pred_prob, pos_label=1)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

print('ROC_AUC:', roc_auc_score(y_test, random_forest_y_pred_prob))


# ## AdaBoost

# ### Hyperparameter Tuning

# In[27]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

execute_cell = True

if execute_cell:
  parameters = {'learning_rate': [0.5, 1, 1.5, 2, 5], 'n_estimators': [100, 200, 300]}
  adaboost = AdaBoostClassifier(algorithm='SAMME', random_state=0)

  clf = GridSearchCV(adaboost, parameters, n_jobs=n_cpus)

  clf.fit(X_train_scaled, y_train_res)

  results_df = pd.DataFrame(clf.cv_results_)

  results_df = results_df.sort_values(by=["rank_test_score"])
  results_df = results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("name")
  display(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]])


# ### Training

# In[28]:


from sklearn.ensemble import AdaBoostClassifier

adaboost = AdaBoostClassifier(learning_rate=1.5, n_estimators=300, algorithm='SAMME', random_state=0)
adaboost.fit(X_train_scaled, y_train_res)


# ### Evaluation

# In[29]:


adaboost_y_pred_prob = adaboost.predict_proba(X_test_scaled)[:,1]
adaboost_y_pred = adaboost.predict(X_test_scaled)


# #### Accuracy and Classification Report

# In[30]:


from sklearn.metrics import accuracy_score, classification_report

print('Accuracy score:', accuracy_score(y_test, adaboost_y_pred))

print('Classifcation report:')
print(classification_report(y_test, adaboost_y_pred, digits=4))


# #### Confusion matrix

# In[31]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmat = confusion_matrix(y_test, adaboost_y_pred)

cmatDisplay = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=adaboost.classes_)

cmatDisplay.plot()
plt.show()


# #### ROC

# In[32]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, adaboost_y_pred_prob, pos_label=1)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

print('ROC_AUC:', roc_auc_score(y_test, adaboost_y_pred_prob))


# ## Gradient Boosting

# ### Hyperparameter Tuning

# In[18]:


from sklearn.model_selection import GridSearchCV
import xgboost as xgb

execute_cell = True

if execute_cell:
  parameters = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 300, 500, 1000], 'max_depth': [3, 5, 7]}
  xgboost = xgb.XGBClassifier(objective='binary:logistic', random_state=0)

  clf = GridSearchCV(xgboost, parameters, n_jobs=n_cpus)

  clf.fit(X_train_scaled, y_train_res)

  results_df = pd.DataFrame(clf.cv_results_)

  results_df = results_df.sort_values(by=["rank_test_score"])
  results_df = results_df.set_index(results_df["params"].apply(lambda x: "_".join(str(val) for val in x.values()))).rename_axis("name")
  display(results_df[["params", "rank_test_score", "mean_test_score", "std_test_score"]])


# ### Training

# In[19]:


import xgboost as xgb
from sklearn.metrics import accuracy_score

xgboost = xgb.XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=1000, subsample = 0.8, colsample_bytree = 0.9, reg_alpha = 0.1, reg_lambda = 0.1, objective='binary:logistic', random_state=0)
xgboost.fit(X_train_scaled, y_train_res)



# ### Evaluation

# In[20]:


xgboost_y_pred_prob = xgboost.predict_proba(X_test_scaled)[:,1]
xgboost_y_pred = xgboost.predict(X_test_scaled)


# #### Accuracy and Classification Report

# In[21]:


from sklearn.metrics import accuracy_score, classification_report

print('Accuracy score:', accuracy_score(y_test, xgboost_y_pred))

print('Classifcation report:')
print(classification_report(y_test, xgboost_y_pred, digits=4))


# #### Confusion matrix

# In[22]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cmat = confusion_matrix(y_test, xgboost_y_pred)

cmatDisplay = ConfusionMatrixDisplay(confusion_matrix=cmat, display_labels=xgboost.classes_)

cmatDisplay.plot()
plt.show()


# #### ROC

# In[23]:


from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, xgboost_y_pred_prob, pos_label=1)

plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)

plt.plot([0,1], [0,1], 'k--' )

print('ROC_AUC:', roc_auc_score(y_test, xgboost_y_pred_prob))

