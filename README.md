# Bitcoin Price Prediction using Time Series

Este proyecto busca predecir el valor futuro de Bitcoin utilizando análisis de series temporales. Empleando diversas técnicas de machine learning y deep learning, se intenta alcanzar una alta precisión en las predicciones de precios de Bitcoin.

## Resumen

El objetivo es construir un modelo que pueda predecir los precios de Bitcoin basándose en datos históricos. El proyecto utiliza forecasting de series temporales para identificar patrones y tendencias en los datos, lo que permite realizar predicciones de valores futuros.

## Instalación con Docker

Para simplificar la implementación y garantizar la portabilidad, el proyecto está diseñado para ejecutarse en contenedores Docker. FastAPI actúa como el servidor principal que expone los endpoints para realizar predicciones y evaluaciones del modelo.

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/lucascarmu/BTC-Price-Predict.git
   cd BTC-Price-Predict
   ```
2. Ejecutar el proyecto con Docker Compose:
   Para levantar los servicios de FastAPI y todos los componentes necesarios, utiliza:
   ```bash
   docker-compose up --build
   ```
3. Endpoints de FastAPI:
   * **/api/predict/** (POST): Realiza predicciones basadas en los días especificados y el nivel de confianza. El POST espera una solicitud con el formato de la clase PredictionRequest:
   ```bash
   {
      "days": 7,
      "confidence_level": 0.95
   }
   ```
   La predicción resultante se almacena en un archivo ensemble_predictions.png en la carpeta outputs, que visualiza las predicciones realizadas junto con los intervalos de confianza.
   * **/api/evaluate/** (GET): Evalúa el rendimiento actual del modelo y devuelve métricas clave de precisión.

### Clase PredictionRequest
La clase PredictionRequest define el formato del POST para el endpoint /api/predict/:
```bash
class PredictionRequest(BaseModel):
    days: int = Field(..., description="Número de días a predecir", ge=1)
    confidence_level: float = Field(1.0, description="Nivel de confianza para la predicción", ge=0, le=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "days": 7,
                "confidence_level": 0.95
            }
        }
    }
```

## Estructura del Proyecto

- `🐋 Dockerfile`               - Archivo de configuración Docker para la imagen del proyecto
- `📂 app/`                     - Código de la aplicación FastAPI
  - `📄 api/endpoints.py`       - Definición de endpoints
  - `🔧 config.py`              - Configuración de la aplicación
  - `📄 main.py`                - Punto de entrada de la aplicación
  - `📄 utils.py`               - Funciones auxiliares
- `📂 data/`                    - Directorio para el conjunto de datos
- `🐋 docker-compose.yml`       - Archivo de configuración para Docker Compose
- `📂 models/`                  - Directorio de modelos entrenados
- `📂 outputs/`                 - Salida de las predicciones y gráficos generados
- `📜 requirements.txt`         - Lista de paquetes requeridos
- `📂 scripts/`                 - Scripts de descarga, preprocesamiento, entrenamiento y evaluación
---

Para cualquier consulta, puedes contactar a: lucascarmusciano@gmail.com.