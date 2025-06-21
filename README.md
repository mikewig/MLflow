# 🧠 Proyecto de Machine Learning con MLflow + Docker

Este proyecto permite entrenar múltiples modelos de clasificación, registrar métricas y parámetros con **MLflow**, y almacenar todo el historial de experimentos de forma organizada y reproducible.

---

## 🚀 Características

- Entrenamiento automático con varios modelos (Logistic Regression, Random Forest, XGBoost, LightGBM, etc.)
- Balanceo de clases con SMOTE
- Registro de métricas en tiempo real en MLflow
- Arquitectura Dockerizada para fácil despliegue

---

## 🧰 Requisitos

- Docker Desktop instalado
- Python 3.8+
- `make` (opcional, para comandos abreviados)

---

## ⚙️ Estructura del proyecto

```
project-root/
├── data/
│   └── dataset.csv          # Tu dataset personalizado
├── mlruns/                  # Logs de MLflow (se crea automáticamente)
├── mlflow-data/             # Base de datos SQLite para MLflow
├── train_models.py          # Script principal de entrenamiento
├── docker-compose.yml       # Configuración del contenedor MLflow
└── README.md                # Este archivo
```

---

## 📦 1. Levantar MLflow con Docker

Ejecuta en la raíz del proyecto:

```bash
docker-compose up -d
```

Esto levantará el servidor de MLflow en:

📍 [http://localhost:5000](http://localhost:5000)

> **Notas**:
> - Si es la primera vez, Docker descargará las imágenes necesarias.
> - Asegúrate de no tener otro servicio ocupando el puerto `5000`.

---

## 🧪 2. Ejecutar el script de entrenamiento

Primero, asegúrate de tener tu entorno virtual y dependencias:

```bash
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
```

Ejecuta el script:

```bash
python train.py
```

Esto entrenará múltiples modelos y los registrará automáticamente en MLflow.

---

## 📊 3. Visualizar experimentos

Abre tu navegador en:

👉 [http://localhost:5000](http://localhost:5000)

Ahí podrás ver:

- Todos los runs por modelo
- Métricas comparativas
- Parámetros utilizados
- Descarga de modelos serializados

---

## 🧹 4. Apagar MLflow

Cuando termines:

```bash
docker-compose down
```

---
