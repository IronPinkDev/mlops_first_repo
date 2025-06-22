import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris

# Importación de los modelos
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

#Carga del conjunto de datos de Iris
iris = load_iris()
X = iris.data
y = iris.target

# Nombres de las clases para el reporte de clasificación
target_names = iris.target_names

#División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Creación de los pipelines para cada modelo para estandarizar el dataset y configurar el modelo

# Pipeline para XGBoost
pipeline_xgb = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False))
])

# Pipeline para CatBoost
pipeline_catboost = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', cb.CatBoostClassifier(verbose=0, loss_function='MultiClass'))
])

# Pipeline para LightGBM
pipeline_lgbm = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', lgb.LGBMClassifier(objective='multiclass'))
])

#Entrenamiento de los modelos utilizando los pipelines
print("Entrenando el modelo XGBoost...")
pipeline_xgb.fit(X_train, y_train)

print("Entrenando el modelo CatBoost...")
pipeline_catboost.fit(X_train, y_train)

print("Entrenando el modelo LightGBM...")
pipeline_lgbm.fit(X_train, y_train)

#Evaluación de los modelos en el conjunto de prueba
print("\n--- Resultados de la Evaluación ---\n")

# Evaluación de XGBoost
y_pred_xgb = pipeline_xgb.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("Modelo: XGBoost")
print(f"Accuracy: {accuracy_xgb:.4f}")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_xgb, target_names=target_names))
print("-" * 40)

# Evaluación de CatBoost
y_pred_catboost = pipeline_catboost.predict(X_test)
accuracy_catboost = accuracy_score(y_test, y_pred_catboost)
print("Modelo: CatBoost")
print(f"Accuracy: {accuracy_catboost:.4f}")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_catboost, target_names=target_names))
print("-" * 40)

# Evaluación de LightGBM
y_pred_lgbm = pipeline_lgbm.predict(X_test)
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
print("Modelo: LightGBM")
print(f"Accuracy: {accuracy_lgbm:.4f}")
print("Reporte de Clasificación:")
print(classification_report(y_test, y_pred_lgbm, target_names=target_names))
print("-" * 40)

#Resumen de resultados
print("\n--- Resumen de Accuracy ---")
print(f"XGBoost:  {accuracy_xgb:.4f}")
print(f"CatBoost: {accuracy_catboost:.4f}")
print(f"LightGBM: {accuracy_lgbm:.4f}")
