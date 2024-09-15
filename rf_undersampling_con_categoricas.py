import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

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

# Sample a portion of the training data (optional)
train_data = train_data.sample(frac=1/5)

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

# Define pipeline with RandomForestClassifier
pipeline = make_pipeline(preprocessor, RandomForestClassifier(random_state=22))

# Define parameters for RandomizedSearchCV
parameters = {
    'randomforestclassifier__n_estimators': np.arange(100, 1000, 100),
    'randomforestclassifier__max_depth': np.arange(5, 50, 5),
    'randomforestclassifier__min_samples_split': np.arange(2, 15),
    'randomforestclassifier__min_samples_leaf': np.arange(1, 15),
    'randomforestclassifier__criterion': ['gini', 'entropy'],
    'randomforestclassifier__bootstrap': [True, False]
}

# Define RandomizedSearchCV with ROC-AUC as the optimization criterion
rs = RandomizedSearchCV(estimator=pipeline,
                        param_distributions=parameters,
                        n_iter=50,  # Aumentar el número de combinaciones a explorar
                        cv=StratifiedKFold(4),  # Validación cruzada estratificada
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
submission_df.to_csv("optimized_model_with_random_forest_and_balancing_and_categorical.csv", sep=",", index=False)
