# 🎓 Guía Práctica U3 - Machine Learning
## Análisis de Rendimiento Académico con Modelos Supervisados y No Supervisados

---

## 📋 Descripción del Proyecto

Este proyecto implementa dos tipos de modelos de Machine Learning para analizar el rendimiento académico de estudiantes:

1. **Modelo Supervisado (Clasificación):** Predice si un estudiante aprobará o reprobará una asignatura
2. **Modelo No Supervisado (Clustering):** Agrupa estudiantes según patrones en su rendimiento

---

## 🎯 Objetivos de Aprendizaje

✅ Preparar y limpiar un dataset real para Machine Learning  
✅ Entrenar un clasificador supervisado usando scikit-learn  
✅ Implementar un modelo de clustering (K-means)  
✅ Visualizar métricas, predicciones y clusters  
✅ Interpretar resultados y presentar conclusiones  

---

## 📁 Estructura del Proyecto

```
GuiaPracticaU3/
│
├── academic_performance_master.csv    # Dataset consolidado
├── analisis_machine_learning.ipynb   # Notebook completo con análisis
├── api.py                             # API REST con FastAPI
├── api_client.py                      # Cliente para probar la API
├── streamlit_app.py                   # Aplicación web interactiva
├── streamlit_app_con_api.py          # Streamlit que consume la API
├── requirements.txt                   # Dependencias del proyecto
└── README.md                          # Este archivo
```

---

## 🚀 Instalación y Configuración

### 1. Requisitos Previos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### 2. Instalar Dependencias

```bash
# Navegar a la carpeta del proyecto
cd "c:\Users\Usuario\OneDrive\Desktop\Universidad\6to Ciclo\Inteligencia Artificial\GuiaPracticaU3"

# Instalar las librerías necesarias
pip install -r requirements.txt
```

---

## 📊 Uso del Proyecto

### 1. Notebook Jupyter

Abre el notebook en VS Code o Jupyter:

```bash
# Si usas Jupyter Notebook
jupyter notebook analisis_machine_learning.ipynb

# Si usas VS Code, simplemente abre el archivo .ipynb
```

El notebook contiene:
- ✅ Exploración detallada de datos (EDA)
- ✅ Modelo supervisado (Árbol de Decisión ⭐, Random Forest, Regresión Logística)
- ✅ Modelo no supervisado (K-Means)
- ✅ Visualizaciones interactivas
- ✅ Interpretación de resultados

### 2. Aplicación Streamlit (Sin API)

```bash
streamlit run streamlit_app.py
```

La aplicación incluye 4 secciones:
1. **Exploración de Datos:** Vista general, distribuciones y relaciones
2. **Modelo Supervisado:** Entrenar clasificadores y ver métricas
3. **Modelo No Supervisado:** Ejecutar clustering y analizar grupos
4. **Comparación de Modelos:** Análisis comparativo

### 3. API REST con FastAPI (Bonus)

```bash
# Iniciar el servidor API
uvicorn api:app --reload

# O ejecutar directamente con Python
python api.py
```

La API incluye endpoints:
- `GET /health` - Verificar estado del servidor
- `POST /api/v1/predict` - Predicción individual
- `POST /api/v1/predict/batch` - Predicción en lote
- `POST /api/v1/cluster` - Clustering de estudiantes
- `GET /api/v1/stats` - Estadísticas del dataset
- `GET /api/v1/niveles` - Niveles académicos disponibles

### 4. Aplicación Streamlit con API

```bash
# Primero inicia la API (en otra terminal)
uvicorn api:app --reload

# Luego ejecuta la aplicación Streamlit
streamlit run streamlit_app_con_api.py
```

Esta versión consume la API en tiempo real para todas las operaciones.

### 5. Cliente de Prueba de API

```bash
python api_client.py
```

Menú interactivo para probar todos los endpoints de la API.

---

## 📈 Modelos Implementados

### 🎓 Modelo Supervisado (Clasificación)

**Objetivo:** Predecir si un estudiante aprobará (1) o reprobará (0)

**Algoritmos:**
- Árbol de Decisión ⭐ (Mejor performance - 93.31%)
- Random Forest (93.24%)
- Regresión Logística (91.05%)

**Características utilizadas:**
- Asistencia (%)
- Número de matrícula
- Nivel académico (codificado)

**Métricas de evaluación:**
- Accuracy
- Precision, Recall, F1-Score
- Matriz de confusión
- Importancia de características

### 🔗 Modelo No Supervisado (Clustering)

**Objetivo:** Agrupar estudiantes según patrones de rendimiento

**Algoritmo:**
- K-Means Clustering

**Características utilizadas:**
- Asistencia promedio (%)
- Nota promedio

**Métricas de evaluación:**
- Silhouette Score
- Método del Codo
- Distribución de clusters

**Perfiles identificados:**
- Estudiantes Excelentes ⭐
- Estudiantes Regulares 📊
- Estudiantes en Riesgo ⚠️

---

## 🏆 Resultados Obtenidos

### Modelo Supervisado
- **Accuracy:** 93.31% (Árbol de Decisión)
- **Característica más importante:** Asistencia
- **Interpretación:** Alta asistencia = Mayor probabilidad de aprobar

### Modelo No Supervisado
- **K óptimo:** 3 clusters
- **Silhouette Score:** 0.6342
- **Interpretación:** Se identifican claramente 3 grupos: Excelentes, Regulares y En Riesgo

---

## 📦 Entregables

Para completar la práctica, debes entregar:

1. ✅ **Notebook (.ipynb)** con desarrollo completo
2. ✅ **Aplicación Streamlit (.py)** con interfaz interactiva
3. ✅ **API REST (Bonus)** - Integración con FastAPI
4. ✅ **Capturas de pantalla:**
   - Entrenamiento del modelo supervisado
   - Métricas y matriz de confusión (Árbol de Decisión)
   - Gráficos del clustering (K=3)
   - Interfaz de Streamlit funcionando
   - API en funcionamiento (si aplica)
5. ✅ **Informe en PDF:**
   - Objetivo
   - Metodología
   - Capturas relevantes con interpretación
   - Resultados obtenidos (Accuracy: 93.31%, Silhouette: 0.6342)
   - Conclusiones y recomendaciones

---

✅ **Implementados en este proyecto:**
- ✨ **Integración con API REST** - FastAPI con endpoints para predicciones y clustering
- ✨ **Aplicación Streamlit con API** - Consume la API en tiempo real
- ✨ **Cliente de prueba de API** - Script interactivo para probar todos los endpoints
- ✨ **Análisis de importancia de características** - Visualización incluida en el notebook
- ✨ **Dashboard con múltiples visualizaciones** - Gráficos interactivos y análisis detallado

**Adicionales opcionales:**
- ✨ Despliegue en la nube (Streamlit Cloud / Railway)
- ✨ Modelos adicionales (SVM, Gradient Boosting)
- ✨ Sistema de autenticación en la API

---

## 🐛 Solución de Problemas

### Error: "No such file or directory"
**Solución:** Asegúrate de que el archivo `academic_performance_master.csv` existe en la carpeta base del proyecto

### Error: "Module not found"
**Solución:** Ejecuta `pip install -r requirements.txt`

### Streamlit no se abre
**Solución:** Verifica que el puerto 8501 no esté ocupado o ejecuta:
```bash
streamlit run streamlit_app.py --server.port 8502
```

### La API no inicia
**Solución:** 
- Verifica que el puerto 8000 esté libre
- Instala uvicorn: `pip install uvicorn`
- Ejecuta: `uvicorn api:app --host 0.0.0.0 --port 8000`

### El modelo tarda mucho
**Solución:** Reduce los parámetros del modelo (menos estimadores en Random Forest) o usa un subset de datos para pruebas

---


## 📄 Licencia

Este proyecto es de uso académico para la Guía Práctica U3.

---
