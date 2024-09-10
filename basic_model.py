import pandas as pd
import gc
import os
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

# Print current working directory
print(os.getcwd())

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Load data
train_data = pd.read_csv("C:/Users/qbcq/OneDrive - Chevron/Desktop/Facultad/TD6/TP2-TD6/TP2_Datos/ctr_21.csv")
eval_data = pd.read_csv("C:/Users/qbcq/OneDrive - Chevron/Desktop/Facultad/TD6/TP2-TD6/TP2_Datos/ctr_test.csv")

# Sample and prepare training data
train_data = train_data.sample(frac=0.1)
y = train_data["Label"]
X = train_data.drop(columns=["Label"]).select_dtypes(include='number')
del train_data
gc.collect()

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=2345)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=2345)

# Function to train and evaluate a model
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_preds)
    roc_auc = roc_auc_score(y_val, y_preds)
    f1 = f1_score(y_val, y_preds)
    return accuracy, roc_auc, f1

# Train and evaluate Decision Tree model
cls_tree = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=8, random_state=2345))
accuracy_tree, roc_auc_tree, f1_tree = train_and_evaluate_model(cls_tree, X_train, y_train, X_val, y_val)
print(f"Decision Tree - Accuracy: {accuracy_tree}, ROC AUC: {roc_auc_tree}, F1 Score: {f1_tree}")

# Train and evaluate XGBoost model
cls_xgb = make_pipeline(SimpleImputer(), XGBClassifier(max_depth=8, random_state=2345, eval_metric='logloss'))
accuracy_xgb, roc_auc_xgb, f1_xgb = train_and_evaluate_model(cls_xgb, X_train, y_train, X_val, y_val)
print(f"XGBoost - Accuracy: {accuracy_xgb}, ROC AUC: {roc_auc_xgb}, F1 Score: {f1_xgb}")

# Train and evaluate Random Forest model
cls_rf = make_pipeline(SimpleImputer(), RandomForestClassifier(max_depth=8, random_state=2345))
accuracy_rf, roc_auc_rf, f1_rf = train_and_evaluate_model(cls_rf, X_train, y_train, X_val, y_val)
print(f"Random Forest - Accuracy: {accuracy_rf}, ROC AUC: {roc_auc_rf}, F1 Score: {f1_rf}")

# Final evaluation on the test set
def final_evaluation(model, X_test, y_test):
    y_preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_preds)
    roc_auc = roc_auc_score(y_test, y_preds)
    f1 = f1_score(y_test, y_preds)
    return accuracy, roc_auc, f1

# Evaluate Decision Tree model on the test set
accuracy_tree_test, roc_auc_tree_test, f1_tree_test = final_evaluation(cls_tree, X_test, y_test)
print(f"Decision Tree Test - Accuracy: {accuracy_tree_test}, ROC AUC: {roc_auc_tree_test}, F1 Score: {f1_tree_test}")

# Evaluate XGBoost model on the test set
accuracy_xgb_test, roc_auc_xgb_test, f1_xgb_test = final_evaluation(cls_xgb, X_test, y_test)
print(f"XGBoost Test - Accuracy: {accuracy_xgb_test}, ROC AUC: {roc_auc_xgb_test}, F1 Score: {f1_xgb_test}")

# Evaluate Random Forest model on the test set
accuracy_rf_test, roc_auc_rf_test, f1_rf_test = final_evaluation(cls_rf, X_test, y_test)
print(f"Random Forest Test - Accuracy: {accuracy_rf_test}, ROC AUC: {roc_auc_rf_test}, F1 Score: {f1_rf_test}")

# Make the submission file for XGBoost model
eval_data = eval_data.select_dtypes(include='number')
eval_data_imputed = SimpleImputer(strategy='mean').fit_transform(eval_data.drop(columns=["id"]))
y_preds_xgb_proba = cls_xgb.predict_proba(eval_data_imputed)[:, 1]
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds_xgb_proba})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("xgboost_model.csv", sep=",", index=False)
