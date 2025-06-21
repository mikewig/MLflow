import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Carga tu dataset desde un archivo CSV
df = pd.read_csv("../data/dataset.csv")  # <-- Cambia esta ruta

# Divide en variables independientes (X) y dependiente (y)
X = df.drop("stroke", axis=1)  # "objetivo" es la columna target
y = df["stroke"]

# Divide en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrena el modelo
model = LinearRegression()
model.fit(X_train, y_train)

# Configura MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("First Experiment")

# Registra el modelo
with mlflow.start_run():
    mlflow.log_param("modelo", "LinearRegression")
    mlflow.log_metric("score", model.score(X_test, y_test))
    mlflow.sklearn.log_model(model, "modelo")