import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

# Cargar los datos
train_data = pd.read_csv("ctr_21.csv")
validation_data_20 = pd.read_csv("ctr_20.csv")
validation_data_16 = pd.read_csv("ctr_16.csv")
validation_data_17 = pd.read_csv("ctr_17.csv")
eval_data = pd.read_csv("ctr_test.csv")

# Muestra opcional de los datos de entrenamiento
train_data = train_data.sample(frac=1/5)

# Separar características y la variable objetivo en el conjunto de entrenamiento
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])

# Tomar el 10% de cada conjunto de validación
validation_data_20 = validation_data_20.sample(frac=0.1, random_state=42)
validation_data_16 = validation_data_16.sample(frac=0.1, random_state=42)
validation_data_17 = validation_data_17.sample(frac=0.1, random_state=42)

# Concatenar los conjuntos de validación
validation_data = pd.concat([validation_data_20, validation_data_16, validation_data_17], ignore_index=True)

# Separar características y la variable objetivo en el conjunto de validación
y_val = validation_data["Label"]
X_val = validation_data.drop(columns=["Label"])

# Liberar memoria
del train_data, validation_data_20, validation_data_16, validation_data_17, validation_data
gc.collect()

# Definir las columnas numéricas y categóricas
num_cols = X_train.select_dtypes(include='number').columns
cat_cols = X_train.select_dtypes(exclude='number').columns

# Preprocesamiento de columnas numéricas y categóricas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(), num_cols),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
    ])

# Definir el pipeline con XGBoost (sin el parámetro 'use_label_encoder')
pipeline = make_pipeline(preprocessor, XGBClassifier(random_state=22))

# Definir los parámetros para RandomizedSearchCV
parameters = {
    'xgbclassifier__n_estimators': np.arange(50, 500, 50),
    'xgbclassifier__max_depth': np.arange(3, 15, 2),
    'xgbclassifier__learning_rate': np.linspace(0.01, 0.3, 10),
    'xgbclassifier__subsample': np.linspace(0.6, 1.0, 5),
    'xgbclassifier__colsample_bytree': np.linspace(0.6, 1.0, 5),
    'xgbclassifier__scale_pos_weight': [1, 3, 5, 10]  # Para balancear las clases
}

# Definir RandomizedSearchCV con ROC-AUC como criterio de optimización
rs = RandomizedSearchCV(estimator=pipeline,
                        param_distributions=parameters,
                        n_iter=20,  # Número de combinaciones a explorar
                        cv=StratifiedKFold(4),  # Validación cruzada estratificada
                        random_state=22,
                        scoring='roc_auc',
                        n_jobs=-1,
                        verbose=1)

# Aplicar submuestreo aleatorio para balancear las clases
X_train_balanced, y_train_balanced = RandomUnderSampler(random_state=22).fit_resample(X_train, y_train)

# Entrenamiento y predicción promediada a través de 10 ejecuciones
val_predictions = np.zeros(X_val.shape[0])
eval_predictions = np.zeros(eval_data.shape[0])

for i in range(10):
    print(f"Entrenamiento y predicción {i+1}/10")

    # Entrenar el modelo
    rs.fit(X_train_balanced, y_train_balanced)

    # Obtener las predicciones en el conjunto de validación
    y_val_pred_proba = rs.best_estimator_.predict_proba(X_val)[:, 1]
    val_predictions += y_val_pred_proba / 10  # Sumar para luego promediar

    # Predecir en el conjunto de evaluación
    eval_data_num = eval_data.select_dtypes(include='number').drop(columns=["id"])
    eval_data_cat = eval_data.select_dtypes(exclude='number')
    
    # Concatenar variables numéricas y categóricas procesadas
    eval_data_preprocessed = pd.concat([eval_data_num, eval_data_cat], axis=1)

    y_eval_pred_proba = rs.best_estimator_.predict_proba(eval_data_preprocessed)[:, 1]
    eval_predictions += y_eval_pred_proba / 10  # Sumar para luego promediar

# Evaluar el promedio de las predicciones en el conjunto de validación
roc_auc_val_avg = roc_auc_score(y_val, val_predictions)
print(f"ROC-AUC promedio en conjunto de validación (10% de ctr_20.csv, ctr_16.csv, ctr_17.csv): {roc_auc_val_avg}")

# Crear el archivo de submission con las predicciones promediadas
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": eval_predictions})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("optimized_model_with_xgboost_averaged_categorical.csv", sep=",", index=False)


