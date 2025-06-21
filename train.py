import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score, roc_auc_score

# Configura MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Second Experiment")

# Carga el dataset
df = pd.read_csv("data/dataset.csv")

# Preprocesamiento básico
X = df.drop("stroke", axis=1)
y = df["stroke"]

# Divide el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify= y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

params = {

     'RandomForestClassifier': {'n_estimators': [135], # Already proved 100 and 50 
                              'max_depth': [15], # 5, 10
                              'min_samples_split': [7], # 5, 3
                              'min_samples_leaf': [6], # 2
                              'max_features': ['sqrt'],
                              'bootstrap': [True],
                              'random_state': [42]},
    
     'KNeighborsClassifier': {'n_neighbors': [7],
                             'weights': ['uniform'],
                             'metric' : ['minkowski']},

     'GradientBoostingClassifier': {'n_estimators': [110], # Already proved 100 and 200 good params 
                                  'learning_rate': [0.5], # 0.1, 0.3
                                  'max_depth': [15], # 5, 10
                                  'min_samples_split': [8], # 5, 8
                                  'min_samples_leaf': [6], # 2, 4
                                  'subsample': [1], # 0.8
                                  'max_features': ['sqrt'],
                                  'random_state': [42]},

      'LGBMClassifier': {'n_estimators': [125], # Already proved 100, 120 and 110
                      'learning_rate': [0.1], # 0.05, 0.08
                      'num_leaves': [31], # 31, 22, 28
                      'max_depth': [-1],
                      'min_child_samples': [20], # 20
                      'subsample': [1], # 0.8, 0.5
                      'colsample_bytree': [1], # 0.8, 0.5
                      'random_state': [42]},

    # 'RidgeClassifier': {'alpha': [1.0],
    #                      'random_state': [42]},

     'XGBClassifier': {'n_estimators': [125], # Already proved 100, 120 and 105
                     'learning_rate': [1], # 0.05, 0.02
                     'max_depth': [16], # 5, 2, 8
                     'subsample': [1], # 0.8, 0.6, 1
                     'colsample_bytree': [1], # 0.8, 0.6, 1
                     'gamma': [0],
                     'reg_alpha': [0.1], # 0.1, 0.05
                     'reg_lambda': [1],
                     'use_label_encoder': [False],
                     'eval_metric': ['logloss'],
                     'random_state': [42]},

    # 'GaussianNB': {},
}

# Modelos a probar
modelos = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
    "KNeighbors": KNeighborsClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

resultados = []

# Entrena y registra cada modelo
for nombre_modelo, modelo in modelos.items():
    grid = params.get(nombre_modelo, [{}])
    for param_set in ParameterGrid(grid):
        print(f"\nEntrenando {nombre_modelo} con parámetros: {param_set}")
        modelo.set_params(**param_set)
        modelo.fit(X_train, y_train)

        y_train_pred = modelo.predict(X_train_res)
        y_test_pred = modelo.predict(X_test)

        acc_train = accuracy_score(y_train_res, y_train_pred)
        acc_test = accuracy_score(y_test, y_test_pred)

        f1_train = f1_score(y_train_res, y_train_pred, average='weighted')
        f1_test = f1_score(y_test, y_test_pred, average='weighted')

        acc_gap = acc_train - acc_test
        f1_gap = f1_train - f1_test

        report = classification_report(y_test, y_test_pred, output_dict=True)

        recall_clase_0 = report['0.0']['recall']
        recall_clase_1 = report['1.0']['recall']

        resultados.append({
            'Modelo': nombre_modelo,
            'Accuracy (train)': round(acc_train, 4),
            'Accuracy (test)': round(acc_test, 4),
            'Overfitting (acc)': round(acc_gap, 4),
            'F1-score (train)': round(f1_train, 4),
            'F1-score (test)': round(f1_test, 4),
            'Overfitting (f1)': round(f1_gap, 4),
        })

        # MLflow logging
        with mlflow.start_run(run_name=f"{nombre_modelo}"):
            mlflow.log_param("modelo", nombre_modelo)
            mlflow.log_params(param_set)
            mlflow.log_metric("accuracy_train", acc_train)
            mlflow.log_metric("accuracy_test", acc_test)
            mlflow.log_metric("f1_train", f1_train)
            mlflow.log_metric("f1_test", f1_test)
            mlflow.log_metric("acc_gap", acc_gap)
            mlflow.log_metric("f1_gap", f1_gap)
            mlflow.log_metric("recall", recall_score(y_train_res, y_train_pred, average='weighted'))
            mlflow.log_metric("recall_0", recall_clase_0)
            mlflow.log_metric("recall_1", recall_clase_1)

            # Registrar modelo en MLflow (automáticamente con sklearn)
            mlflow.sklearn.log_model(modelo, "modelo")
