import pandas as pd
import gc
import os
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split

"""print(os.getcwd())

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Load data
train_data = pd.read_csv("C:/Users/qbcq/OneDrive - Chevron/Desktop/Facultad/TD6/TP2-TD6/TP2_Datos/ctr_21.csv")
eval_data = pd.read_csv("C:/Users/qbcq/OneDrive - Chevron/Desktop/Facultad/TD6/TP2-TD6/TP2_Datos/ctr_test.csv")

# Sample and prepare training data
train_data = train_data.sample(frac=1/10)
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])
X_train = X_train.select_dtypes(include='number')
del train_data
gc.collect()

# Prepare evaluation data
eval_data = eval_data.select_dtypes(include='number')
y_true = eval_data["Label"]

# Train Decision Tree model
cls_tree = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=2345))
cls_tree.fit(X_train, y_train)

# Predict with Decision Tree model
y_preds_tree = cls_tree.predict(eval_data.drop(columns=["id"]))
accuracy_tree = accuracy_score(y_true, y_preds_tree)
roc_auc_tree = roc_auc_score(y_true, y_preds_tree)
f1_tree = f1_score(y_true, y_preds_tree)

print(f"Decision Tree - Accuracy: {accuracy_tree}, ROC AUC: {roc_auc_tree}, F1 Score: {f1_tree}")

# Train XGBoost model
cls_xgb = make_pipeline(SimpleImputer(), XGBClassifier(max_depth=8, random_state=2345, eval_metric='logloss'))
cls_xgb.fit(X_train, y_train)

# Predict with XGBoost model
y_preds_xgb = cls_xgb.predict(eval_data.drop(columns=["id"]))
accuracy_xgb = accuracy_score(y_true, y_preds_xgb)
roc_auc_xgb = roc_auc_score(y_true, y_preds_xgb)
f1_xgb = f1_score(y_true, y_preds_xgb)

print(f"XGBoost - Accuracy: {accuracy_xgb}, ROC AUC: {roc_auc_xgb}, F1 Score: {f1_xgb}")

# Make the submission file for XGBoost model
# y_preds_xgb_proba = cls_xgb.predict_proba(eval_data.drop(columns=["id"]))[:, 1]
# submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds_xgb_proba})
# submission_df["id"] = submission_df["id"].astype(int)
# submission_df.to_csv("xgboost_model.csv", sep=",", index=False)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=2345)

# Train Balanced Random Forest model
cls_rf = make_pipeline(SimpleImputer(), BalancedRandomForestClassifier(max_depth=8, random_state=2345))
cls_rf.fit(X_train, y_train)

# Predict with Balanced Random Forest model
y_preds_rf = cls_rf.predict(X_val)
accuracy_rf = accuracy_score(y_val, y_preds_rf)
roc_auc_rf = roc_auc_score(y_val, y_preds_rf)
f1_rf = f1_score(y_val, y_preds_rf)

print(f"Balanced Random Forest - Accuracy: {accuracy_rf}, ROC AUC: {roc_auc_rf}, F1 Score: {f1_rf}")

# Make the submission file for Balanced Random Forest model
eval_data = pd.read_csv("C:/Users/qbcq/OneDrive - Chevron/Desktop/Facultad/TD6/TP2-TD6/TP2_Datos/ctr_test.csv")
eval_data = eval_data.select_dtypes(include='number')
eval_data_imputed = imputer.transform(eval_data.drop(columns=["id"]))
y_preds_rf_proba = cls_rf.predict_proba(eval_data_imputed)[:, 1]
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds_rf_proba})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("balanced_rf_model.csv", sep=",", index=False)"""


from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier

print(os.getcwd())

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Load data
train_data = pd.read_csv("C:/Users/qbcq/OneDrive - Chevron/Desktop/Facultad/TD6/TP2-TD6/TP2_Datos/ctr_21.csv")

# Sample and prepare training data
train_data = train_data.sample(frac=1/10)
y = train_data["Label"]
X = train_data.drop(columns=["Label"])
X = X.select_dtypes(include='number')
del train_data
gc.collect()

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=2345)

# Train Balanced Random Forest model
cls_rf = make_pipeline(SimpleImputer(), BalancedRandomForestClassifier(max_depth=8, random_state=2345))
cls_rf.fit(X_train, y_train)

# Predict with Balanced Random Forest model
y_preds_rf = cls_rf.predict(X_val)
accuracy_rf = accuracy_score(y_val, y_preds_rf)
roc_auc_rf = roc_auc_score(y_val, y_preds_rf)
f1_rf = f1_score(y_val, y_preds_rf)

print(f"Balanced Random Forest - Accuracy: {accuracy_rf}, ROC AUC: {roc_auc_rf}, F1 Score: {f1_rf}")

# Make the submission file for Balanced Random Forest model
eval_data = pd.read_csv("C:/Users/qbcq/OneDrive - Chevron/Desktop/Facultad/TD6/TP2-TD6/TP2_Datos/ctr_test.csv")
eval_data = eval_data.select_dtypes(include='number')
eval_data_imputed = imputer.transform(eval_data.drop(columns=["id"]))
y_preds_rf_proba = cls_rf.predict_proba(eval_data_imputed)[:, 1]
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds_rf_proba})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("balanced_rf_model.csv", sep=",", index=False)
