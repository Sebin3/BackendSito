# Documentación Completa - API de Machine Learning y Análisis de Datos

## Descripción General

Esta API REST proporciona un sistema completo para gestión de usuarios, análisis de datasets y entrenamiento de modelos de Machine Learning con predicciones de negocio. Construida con Django REST Framework y tecnologías de ML como scikit-learn, XGBoost y PyTorch.

**URL Base:** `http://localhost:8000/api/`

---

## 1. MÓDULO DE AUTENTICACIÓN (USER)

### Descripción
Sistema completo de autenticación con JWT, gestión de perfiles y avatares.

### Endpoints

| Método | Endpoint | Descripción | Autenticación |
|--------|----------|-------------|---------------|
| POST | `/auth/register/` | Registro de nuevo usuario | No |
| POST | `/auth/login/` | Iniciar sesión | No |
| POST | `/auth/logout/` | Cerrar sesión | Sí |
| POST | `/auth/token/refresh/` | Renovar token de acceso | No |
| GET | `/auth/profile/` | Obtener perfil del usuario | Sí |
| PUT | `/auth/profile/` | Actualizar perfil completo | Sí |
| PATCH | `/auth/profile/` | Actualizar perfil parcial | Sí |
| POST | `/auth/profile/avatar/upload/` | Subir avatar | Sí |
| DELETE | `/auth/profile/avatar/delete/` | Eliminar avatar | Sí |

### Ejemplos de Uso

#### 1.1 Registro de Usuario
```http
POST /api/auth/register/
Content-Type: application/json

{
    "email": "usuario@ejemplo.com",
    "first_name": "Juan",
    "last_name": "Pérez",
    "password": "contraseña123",
    "password2": "contraseña123"
}
```

**Respuesta exitosa (201):**
```json
{
    "message": "Usuario registrado exitosamente",
    "user": {
        "id": 1,
        "email": "usuario@ejemplo.com",
        "first_name": "Juan",
        "last_name": "Pérez",
        "username": "usuario@ejemplo.com",
        "avatar": null,
        "avatar_url": null,
        "created_at": "2024-01-15T10:30:00Z",
        "updated_at": "2024-01-15T10:30:00Z"
    },
    "tokens": {
        "refresh": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
        "access": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
    }
}
```

#### 1.2 Iniciar Sesión
```http
POST /api/auth/login/
Content-Type: application/json

{
    "email": "usuario@ejemplo.com",
    "password": "contraseña123"
}
```

#### 1.3 Obtener Perfil
```http
GET /api/auth/profile/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

#### 1.4 Subir Avatar
```http
POST /api/auth/profile/avatar/upload/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Content-Type: multipart/form-data

avatar: [archivo de imagen]
```

---

## 2. MÓDULO DE DATASETS

### Descripción
Gestión completa de datasets con análisis estadístico, limpieza de datos y visualización.

### Endpoints

| Método | Endpoint | Descripción | Autenticación |
|--------|----------|-------------|---------------|
| GET | `/data/datasets/` | Listar datasets del usuario | Sí |
| POST | `/data/datasets/` | Subir nuevo dataset | Sí |
| GET | `/data/datasets/{id}/` | Obtener dataset específico | Sí |
| PUT | `/data/datasets/{id}/` | Actualizar dataset | Sí |
| DELETE | `/data/datasets/{id}/` | Eliminar dataset | Sí |
| GET | `/data/datasets/{id}/preview/` | Vista previa del dataset | Sí |
| GET | `/data/datasets/{id}/stats/` | Estadísticas detalladas | Sí |
| POST | `/data/datasets/{id}/clean/` | Limpiar dataset | Sí |
| GET | `/data/datasets/{id}/download/` | Descargar dataset | Sí |
| GET | `/data/cleaned-datasets/` | Listar datasets limpios | Sí |
| GET | `/data/cleaned-datasets/{id}/` | Obtener dataset limpio | Sí |
| GET | `/data/cleaned-datasets/{id}/preview/` | Vista previa dataset limpio | Sí |
| GET | `/data/cleaned-datasets/{id}/download/` | Descargar dataset limpio | Sí |

### Ejemplos de Uso

#### 2.1 Subir Dataset
```http
POST /api/data/datasets/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Content-Type: multipart/form-data

name: "Ventas 2024"
description: "Dataset de ventas del año 2024"
file: [archivo CSV]
```

**Respuesta exitosa (201):**
```json
{
    "id": 1,
    "name": "Ventas 2024",
    "description": "Dataset de ventas del año 2024",
    "file": "/media/datasets/dataset_1_Ventas_2024.csv",
    "file_url": "http://localhost:8000/media/datasets/dataset_1_Ventas_2024.csv",
    "owner": 1,
    "owner_email": "usuario@ejemplo.com",
    "file_size": 1048576,
    "rows_count": 1000,
    "columns_count": 8,
    "columns_info": {
        "fecha": {
            "dtype": "object",
            "null_count": 0,
            "unique_count": 365,
            "sample_values": ["2024-01-01", "2024-01-02", "2024-01-03"]
        },
        "ventas": {
            "dtype": "float64",
            "null_count": 5,
            "unique_count": 995,
            "sample_values": [1500.50, 2300.75, 1800.25]
        }
    },
    "null_values_count": 15,
    "duplicate_rows_count": 2,
    "data_quality_score": 98.3,
    "created_at": "2024-01-15T10:30:00Z",
    "updated_at": "2024-01-15T10:30:00Z"
}
```

#### 2.2 Vista Previa del Dataset
```http
GET /api/data/datasets/1/preview/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

**Respuesta:**
```json
{
    "columns": ["fecha", "producto", "ventas", "cantidad", "precio_unitario"],
    "data": [
        {
            "fecha": "2024-01-01",
            "producto": "Laptop",
            "ventas": 1500.50,
            "cantidad": 1,
            "precio_unitario": 1500.50
        }
    ],
    "total_rows": 1000,
    "showing_rows": 20,
    "data_types": {
        "fecha": "object",
        "ventas": "float64",
        "cantidad": "int64"
    },
    "missing_values": {
        "fecha": 0,
        "ventas": 5,
        "cantidad": 0
    },
    "basic_stats": {
        "ventas": {
            "mean": 1850.25,
            "median": 1800.00,
            "std": 450.30,
            "min": 500.00,
            "max": 5000.00
        }
    }
}
```

#### 2.3 Estadísticas Detalladas
```http
GET /api/data/datasets/1/stats/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

#### 2.4 Limpiar Dataset

**Endpoint:**
```http
POST /api/data/datasets/{id}/clean/
Authorization: Bearer {token}
Content-Type: application/json
```

**Ejemplo 1 - Limpieza Básica (Recomendada):**
```json
{
    "null_strategy": "fill_median",
    "target_columns": [],
    "remove_duplicates": true,
    "convert_data_types": true,
    "trim_whitespace": true
}
```

**Ejemplo 2 - Limpieza para Machine Learning:**
```json
{
    "null_strategy": "fill_median",
    "target_columns": [],
    "remove_duplicates": true,
    "remove_outliers": true,
    "outlier_method": "iqr",
    "standardize_data": true,
    "convert_data_types": true,
    "trim_whitespace": true
}
```

**Ejemplo 3 - Limpieza con Valores Personalizados:**
```json
{
    "null_strategy": "fill_median",
    "target_columns": [],
    "custom_fill_values": {
        "nombre": "Sin especificar",
        "departamento": "Sin asignar"
    },
    "remove_duplicates": true,
    "convert_data_types": true,
    "trim_whitespace": true
}
```

**Ejemplo 4 - Limpieza Agresiva (Eliminar nulos):**
```json
{
    "null_strategy": "drop",
    "target_columns": [],
    "remove_duplicates": true,
    "remove_outliers": true,
    "convert_data_types": true
}
```

---

### **Opciones de Limpieza Disponibles:**

#### **null_strategy (Obligatorio):**
- `drop` - Eliminar filas con valores nulos
- `fill_mean` - Rellenar con media (promedio)
- `fill_median` - Rellenar con mediana (valor del medio)
- `fill_mode` - Rellenar con moda (más frecuente)
- `fill_forward` - Rellenar con valor anterior
- `fill_backward` - Rellenar con valor siguiente
- `fill_interpolate` - Interpolación lineal
- `fill_zero` - Rellenar con cero
- `fill_custom` - Valor personalizado

#### **target_columns (Opcional):**
- Array de columnas específicas: `["edad", "salario"]`
- Vacío `[]` para aplicar a todas las columnas

#### **custom_fill_value (Opcional):**
- Valor único para todas las columnas: `"Sin especificar"`
- Solo con `fill_custom`

#### **custom_fill_values (Opcional):**
- Valores específicos por columna: `{"nombre": "Sin nombre", "ciudad": "N/A"}`
- Funciona con `fill_median` y `fill_custom`
- Sobrescribe la estrategia general para columnas específicas

#### **remove_duplicates (Opcional):**
- `true` - Eliminar filas duplicadas
- `false` - Mantener duplicados

#### **remove_outliers (Opcional):**
- `true` - Eliminar valores atípicos
- `false` - Mantener todos los valores

#### **outlier_method (Opcional):**
- `iqr` - Rango Intercuartílico (recomendado)
- `zscore` - Z-Score (más agresivo)
- `isolation_forest` - Machine Learning

#### **normalize_data (Opcional):**
- `true` - Escalar valores entre 0 y 1
- `false` - Mantener valores originales

#### **standardize_data (Opcional):**
- `true` - Estandarizar (media=0, desviación=1)
- `false` - Mantener valores originales

#### **numeric_columns (Opcional):**
- Array de columnas numéricas para normalizar/estandarizar
- Ejemplo: `["edad", "salario", "experiencia"]`

#### **remove_empty_rows (Opcional):**
- `true` - Eliminar filas completamente vacías
- `false` - Mantener filas vacías

#### **remove_empty_columns (Opcional):**
- `true` - Eliminar columnas completamente vacías
- `false` - Mantener columnas vacías

#### **convert_data_types (Opcional):**
- `true` - Convertir texto a números automáticamente
- `false` - Mantener tipos originales

#### **trim_whitespace (Opcional):**
- `true` - Eliminar espacios extra en textos
- `false` - Mantener espacios

---

### **Respuesta de Limpieza:**
```json
{
    "message": "Dataset limpiado exitosamente",
    "cleaned_dataset": {
        "id": 2,
        "name": "Ventas_2024_limpio_1",
        "rows_count": 148,
        "columns_count": 12,
        "rows_removed": 2,
        "null_values_filled": 15,
        "file_url": "/media/datasets/cleaned/cleaned_2.csv"
    },
    "cleaning_report": {
        "methods_used": [
            "Eliminar espacios en blanco",
            "Conversión automática de tipos",
            "Rellenar con mediana/moda",
            "Eliminar duplicados"
        ],
        "null_values_filled": 15,
        "rows_removed": 2,
        "outliers_removed": 0,
        "columns_removed": 0
    }
}
```

---

## 3. MÓDULO DE MACHINE LEARNING

### Descripción
Sistema completo de entrenamiento de modelos ML con soporte para Random Forest, XGBoost y Neural Networks (PyTorch).

### Endpoints

| Método | Endpoint | Descripción | Autenticación |
|--------|----------|-------------|---------------|
| GET | `/ml/models/` | Listar modelos del usuario | Sí |
| POST | `/ml/models/train/` | Entrenar nuevo modelo | Sí |
| GET | `/ml/models/{id}/` | Obtener modelo específico | Sí |
| GET | `/ml/models/{id}/metrics/` | Métricas detalladas del modelo | Sí |
| GET | `/ml/models/dataset_info/` | Info de columnas de dataset | Sí |
| GET | `/ml/predictions/` | Listar predicciones | Sí |
| POST | `/ml/predictions/predict/` | Realizar predicción individual | Sí |
| POST | `/ml/predictions/batch_predict/` | Predicciones en lote | Sí |
| GET | `/ml/comparisons/` | Listar comparaciones | Sí |
| POST | `/ml/comparisons/` | Crear comparación | Sí |
| POST | `/ml/comparisons/{id}/compare/` | Ejecutar comparación | Sí |

### Endpoints de Predicciones de Negocio

| Método | Endpoint | Descripción | Autenticación |
|--------|----------|-------------|---------------|
| POST | `/ml/business-predictions/revenue_forecast/` | Predicción de ganancias futuras | Sí |
| POST | `/ml/business-predictions/promotion_impact/` | Impacto de promociones | Sí |
| POST | `/ml/business-predictions/seasonal_trends/` | Tendencias estacionales | Sí |
| POST | `/ml/business-predictions/growth_scenarios/` | Escenarios de crecimiento | Sí |
| POST | `/ml/business-predictions/quick_forecast/` | Predicción rápida | Sí |

### Ejemplos de Uso

#### 3.1 Entrenar Modelo
```http
POST /api/ml/models/train/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Content-Type: application/json

{
    "dataset_type": "cleaned",
    "dataset_id": 1,
    "algorithm": "random_forest",
    "task_type": "regression",
    "target_column": "ventas",
    "feature_columns": ["cantidad", "precio_unitario", "mes", "dia_semana"],
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 42,
    "hyperparameters": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2
    }
}
```

**Algoritmos disponibles:**
- `random_forest`: Random Forest (clasificación/regresión)
- `gradient_boosting`: XGBoost (clasificación/regresión)
- `neural_network`: Red Neuronal con PyTorch

#### 3.2 Realizar Predicción
```http
POST /api/ml/predictions/predict/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Content-Type: application/json

{
    "model_id": 1,
    "input_data": {
        "cantidad": 2,
        "precio_unitario": 1200.00,
        "mes": 12,
        "dia_semana": 5
    }
}
```

**Respuesta:**
```json
{
    "id": 1,
    "ml_model": 1,
    "ml_model_name": "random_forest_Ventas_2024_1642089600",
    "input_data": {
        "cantidad": 2,
        "precio_unitario": 1200.00,
        "mes": 12,
        "dia_semana": 5
    },
    "prediction_result": {
        "prediction": 2400.50
    },
    "confidence_score": 0.85,
    "prediction_time": 0.023,
    "created_at": "2024-01-15T15:30:00Z"
}
```

#### 3.3 Predicción de Ganancias Futuras
```http
POST /api/ml/business-predictions/revenue_forecast/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Content-Type: application/json

{
    "ventas_historicas": [15000, 18000, 16500, 19200, 17800],
    "costos_promedio": 0.6,
    "tendencia": "creciente",
    "estacionalidad": 1.2,
    "promociones_planeadas": 2
}
```

**Respuesta:**
```json
{
    "tipo_prediccion": "revenue_forecast",
    "fecha_prediccion": "2024-01-15T15:30:00Z",
    "predicciones": {
        "7_dias": {
            "ventas_estimadas": 22560.00,
            "costos_estimados": 13536.00,
            "ganancia_neta": 9024.00,
            "margen_ganancia": 40.0
        },
        "30_dias": {
            "ventas_estimadas": 98112.50,
            "costos_estimados": 58867.50,
            "ganancia_neta": 39245.00,
            "margen_ganancia": 40.0
        },
        "90_dias": {
            "ventas_estimadas": 301250.75,
            "costos_estimados": 180750.45,
            "ganancia_neta": 120500.30,
            "margen_ganancia": 40.0
        },
        "365_dias": {
            "ventas_estimadas": 1285000.00,
            "costos_estimados": 771000.00,
            "ganancia_neta": 514000.00,
            "margen_ganancia": 40.0
        }
    },
    "factores_considerados": {
        "venta_base_semanal": 17300.0,
        "tendencia": "creciente",
        "factor_tendencia": 1.15,
        "estacionalidad": 1.2,
        "costos_promedio": 0.6,
        "promociones_planeadas": 2
    },
    "recomendaciones": [
        "Tendencia positiva: Considera invertir en marketing para acelerar el crecimiento",
        "Planifica reinversión del 15-20% de ganancias para sostener el crecimiento",
        "Excelente proyección: Considera expandir a nuevos mercados"
    ]
}
```

#### 3.4 Impacto de Promociones
```http
POST /api/ml/business-predictions/promotion_impact/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Content-Type: application/json

{
    "venta_base_semanal": 10000,
    "descuento": 0.20,
    "duracion_dias": 7,
    "categoria": "electronica",
    "costos_promedio": 0.6
}
```

#### 3.5 Predicción Rápida
```http
POST /api/ml/business-predictions/quick_forecast/
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
Content-Type: application/json

{
    "ganancia_semanal_actual": 5000,
    "dias_prediccion": 30,
    "factor_crecimiento": 1.1
}
```

---

## Códigos de Estado HTTP

| Código | Descripción |
|--------|-------------|
| 200 | OK - Solicitud exitosa |
| 201 | Created - Recurso creado exitosamente |
| 400 | Bad Request - Error en los datos enviados |
| 401 | Unauthorized - Token de autenticación requerido |
| 403 | Forbidden - Sin permisos para acceder |
| 404 | Not Found - Recurso no encontrado |
| 500 | Internal Server Error - Error del servidor |

## Autenticación

La API utiliza JWT (JSON Web Tokens) para autenticación. Incluye el token en el header:

```
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
```

**Duración de tokens:**
- Access Token: 60 minutos
- Refresh Token: 7 días

## Formatos de Archivo Soportados

- **Datasets**: CSV (máximo 50MB)
- **Avatares**: JPG, PNG, GIF (máximo 5MB)
- **Modelos**: Joblib (generado automáticamente)

## Límites y Restricciones

- Máximo 1000 predicciones por lote
- Datasets hasta 50MB
- Avatares hasta 5MB
- Modelos se entrenan de forma síncrona (en producción sería asíncrono)

## Tecnologías Utilizadas

- **Backend**: Django 5.2.7, Django REST Framework
- **Autenticación**: JWT con djangorestframework-simplejwt
- **Machine Learning**: scikit-learn, XGBoost, PyTorch
- **Análisis de Datos**: pandas, numpy, scipy
- **Base de Datos**: SQLite (desarrollo)
- **Archivos**: Sistema de archivos local con Django Media

## Instalación y Configuración

1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar migraciones: `python manage.py migrate`
3. Crear superusuario: `python manage.py createsuperuser`
4. Ejecutar servidor: `python manage.py runserver`

La API estará disponible en `http://localhost:8000/api/`