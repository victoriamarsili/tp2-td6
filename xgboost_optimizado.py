import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from xgboost import XGBClassifier
import joblib

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Load training data from 'ctr_21.csv'
train_data = pd.read_csv("ctr_21.csv")

# Load validation data from 'ctr_20.csv', 'ctr_16.csv', and 'ctr_17.csv'
validation_data_20 = pd.read_csv("ctr_20.csv")
validation_data_16 = pd.read_csv("ctr_16.csv")
validation_data_17 = pd.read_csv("ctr_17.csv")

# Load the evaluation (test) data from 'ctr_test.csv'
eval_data = pd.read_csv("ctr_test.csv")

# Separate features and target variable for training data
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])

# Sample 10% from each validation dataset
validation_data_20 = validation_data_20.sample(frac=0.1, random_state=42)
validation_data_16 = validation_data_16.sample(frac=0.1, random_state=42)
validation_data_17 = validation_data_17.sample(frac=0.1, random_state=42)

# Concatenate the three validation datasets into one
validation_data = pd.concat([validation_data_20, validation_data_16, validation_data_17], ignore_index=True)

# Separate features and target variable for validation data
y_val = validation_data["Label"]
X_val = validation_data.drop(columns=["Label"])

# Free up memory
del train_data, validation_data_20, validation_data_16, validation_data_17, validation_data
gc.collect()

# Define columns
numeric_features = X_train.select_dtypes(include='number').columns
categorical_features = X_train.select_dtypes(exclude='number').columns

# Preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), numeric_features),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features)
    ])

# Define pipeline with XGBClassifier
pipeline = make_pipeline(preprocessor, XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=22))

# Define parameters for RandomizedSearchCV
parameters = {
    'xgbclassifier__n_estimators': np.arange(100, 1000, 100),
    'xgbclassifier__max_depth': np.arange(3, 15),  # Ajustar el rango
    'xgbclassifier__min_child_weight': np.arange(1, 10),
    'xgbclassifier__subsample': np.arange(0.5, 1.1, 0.1),  # Submuestreo
    'xgbclassifier__colsample_bytree': np.arange(0.5, 1.1, 0.1),  # Muestreo de columnas
    'xgbclassifier__gamma': [0, 0.1, 0.5, 1],  # Para controlar la regularización
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.2, 0.3]  # Ajustar la tasa de aprendizaje
}

# Define RandomizedSearchCV with ROC-AUC as the optimization criterion
rs = RandomizedSearchCV(estimator=pipeline,
                        param_distributions=parameters,
                        n_iter=50,  # Aumentar el número de combinaciones a explorar
                        cv=StratifiedKFold(5),  # Validación cruzada estratificada
                        random_state=22,
                        scoring='roc_auc',
                        n_jobs=-1,
                        verbose=1)

# Apply RandomUnderSampler for class balancing
X_train_balanced, y_train_balanced = RandomUnderSampler(random_state=22).fit_resample(X_train, y_train)

# Train the optimized model with the balanced training set
rs.fit(X_train_balanced, y_train_balanced)

# Show the best parameters and the best score in training
print(f"Mejores parámetros: {rs.best_params_}")
print(f"Mejor ROC-AUC en validación cruzada: {rs.best_score_}")

# Evaluate the best model on the validation set
y_val_pred_proba = rs.best_estimator_.predict_proba(X_val)[:, 1]
roc_auc_val = roc_auc_score(y_val, y_val_pred_proba)
print(f"ROC-AUC en conjunto de validación (10% de ctr_20.csv, ctr_16.csv, ctr_17.csv): {roc_auc_val}")

# Predict on the evaluation set (for submission)
eval_data_num = eval_data.select_dtypes(include='number').drop(columns=["id"])
eval_data_cat = eval_data.select_dtypes(exclude='number')
eval_data_preprocessed = pd.concat([eval_data_num, eval_data_cat], axis=1)

y_preds = rs.best_estimator_.predict_proba(eval_data_preprocessed)[:, 1]

# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("optimized_model_with_xgboost_and_balancing.csv", sep=",", index=False)

# Save the best model
joblib.dump(rs.best_estimator_, 'best_xgb_model.pkl')


