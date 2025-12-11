"""
API REST para el Sistema de Análisis de Rendimiento Académico
Guía Práctica U3 - Inteligencia Artificial

Esta API proporciona endpoints para:
- Predecir aprobación/reprobación de estudiantes (Modelo Supervisado)
- Obtener clustering de estudiantes (Modelo No Supervisado)
- Consultar estadísticas generales

Tecnologías: FastAPI + scikit-learn

Autor: [Tu Nombre]
Fecha: 11 de Diciembre de 2025
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime

# Scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURACIÓN DE LA API
# ============================================================================

app = FastAPI(
    title="API de Análisis de Rendimiento Académico",
    description="API REST para predicción y clustering de rendimiento estudiantil",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS para permitir peticiones desde Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODELOS PYDANTIC (Esquemas de datos)
# ============================================================================

class EstudianteInput(BaseModel):
    """Datos de entrada para predicción supervisada"""
    asistencia: float = Field(..., ge=0, le=100, description="Porcentaje de asistencia (0-100)")
    num_matricula: int = Field(..., ge=1, description="Número de matrícula (1 o más)")
    nivel: str = Field(..., description="Nivel académico (PRIMERO, SEGUNDO, TERCERO, etc.)")
    
    class Config:
        schema_extra = {
            "example": {
                "asistencia": 85.5,
                "num_matricula": 1,
                "nivel": "PRIMERO"
            }
        }

class PrediccionResponse(BaseModel):
    """Respuesta de predicción"""
    estudiante_id: Optional[str] = None
    prediccion: str = Field(..., description="APROBADO o REPROBADO")
    probabilidad_aprobar: float = Field(..., description="Probabilidad de aprobar (0-1)")
    probabilidad_reprobar: float = Field(..., description="Probabilidad de reprobar (0-1)")
    confianza: str = Field(..., description="ALTA, MEDIA o BAJA")
    recomendaciones: List[str] = Field(..., description="Recomendaciones para el estudiante")
    timestamp: str = Field(..., description="Fecha y hora de la predicción")

class EstudianteClusterInput(BaseModel):
    """Datos de entrada para clustering"""
    asistencia_promedio: float = Field(..., ge=0, le=100, description="Asistencia promedio (%)")
    nota_promedio: float = Field(..., ge=0, le=10, description="Nota promedio (0-10)")
    
    class Config:
        schema_extra = {
            "example": {
                "asistencia_promedio": 85.0,
                "nota_promedio": 7.5
            }
        }

class ClusterResponse(BaseModel):
    """Respuesta de clustering"""
    cluster: int = Field(..., description="Número de cluster asignado")
    perfil: str = Field(..., description="Perfil del cluster")
    descripcion: str = Field(..., description="Descripción del perfil")
    distancia_al_centroide: float = Field(..., description="Distancia al centroide del cluster")
    recomendaciones: List[str] = Field(..., description="Recomendaciones basadas en el cluster")

class EstadisticasResponse(BaseModel):
    """Estadísticas generales del dataset"""
    total_registros: int
    total_estudiantes: int
    proporcion_aprobados: float
    asistencia_promedio: float
    nota_promedio: float
    accuracy_modelo: float
    silhouette_score: float

class HealthResponse(BaseModel):
    """Estado de salud de la API"""
    status: str
    timestamp: str
    modelo_supervisado_cargado: bool
    modelo_clustering_cargado: bool
    dataset_cargado: bool
    total_registros: int

# ============================================================================
# VARIABLES GLOBALES
# ============================================================================

# Modelos y datos
modelo_supervisado = None
modelo_clustering = None
scaler_supervisado = None
scaler_clustering = None
label_encoder = None
df_clean = None
student_df = None

# Métricas
modelo_accuracy = 0.0
clustering_silhouette = 0.0

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def cargar_y_preparar_datos():
    """Carga y prepara los datos al iniciar la API"""
    global df_clean, student_df, label_encoder
    
    print("📂 Cargando dataset...")
    
    try:
        # Cargar dataset
        df = pd.read_csv('academic_performance_master.csv')
        
        # Preparar datos
        columnas_relevantes = [
            'Identificacion_Estudiante',
            'Estudiante',
            'Nivel',
            'Asignatura',
            'Num_matricula',
            'Asistencia',
            'Nota_final',
            'Estado_Asignatura'
        ]
        
        df_clean = df[columnas_relevantes].copy()
        df_clean.dropna(subset=['Asistencia', 'Nota_final', 'Estado_Asignatura'], inplace=True)
        df_clean['Asistencia'] = df_clean['Asistencia'].clip(lower=0, upper=100)
        df_clean['Nota_final'] = df_clean['Nota_final'].clip(lower=0, upper=10)
        df_clean['Aprobado'] = (df_clean['Estado_Asignatura'] == 'APROBADO').astype(int)
        
        # Codificar nivel
        label_encoder = LabelEncoder()
        df_clean['Nivel_encoded'] = label_encoder.fit_transform(df_clean['Nivel'].astype(str))
        
        # Agregar por estudiante
        student_df = df_clean.groupby('Identificacion_Estudiante').agg({
            'Estudiante': 'first',
            'Asistencia': 'mean',
            'Nota_final': 'mean',
            'Num_matricula': 'max'
        }).reset_index()
        
        student_df.columns = [
            'Identificacion_Estudiante',
            'Estudiante',
            'Asistencia_promedio',
            'Nota_promedio',
            'Num_matriculas'
        ]
        
        print(f"✅ Datos cargados: {len(df_clean)} registros, {len(student_df)} estudiantes")
        return True
        
    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        return False

def entrenar_modelo_supervisado():
    """Entrena el modelo supervisado"""
    global modelo_supervisado, scaler_supervisado, modelo_accuracy
    
    print("🎓 Entrenando modelo supervisado...")
    
    try:
        # Preparar datos
        feature_columns = ['Asistencia', 'Num_matricula', 'Nivel_encoded']
        X = df_clean[feature_columns].values
        y = df_clean['Aprobado'].values
        
        # Escalar
        scaler_supervisado = StandardScaler()
        X_scaled = scaler_supervisado.fit_transform(X)
        
        # Dividir
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Entrenar Random Forest
        modelo_supervisado = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=42
        )
        modelo_supervisado.fit(X_train, y_train)
        
        # Calcular accuracy
        modelo_accuracy = modelo_supervisado.score(X_test, y_test)
        
        print(f"✅ Modelo supervisado entrenado. Accuracy: {modelo_accuracy*100:.2f}%")
        return True
        
    except Exception as e:
        print(f"❌ Error al entrenar modelo supervisado: {e}")
        return False

def entrenar_modelo_clustering():
    """Entrena el modelo de clustering"""
    global modelo_clustering, scaler_clustering, clustering_silhouette
    
    print("🔗 Entrenando modelo de clustering...")
    
    try:
        # Preparar datos
        from sklearn.metrics import silhouette_score
        
        X_clustering = student_df[['Asistencia_promedio', 'Nota_promedio']].values
        
        # Escalar
        scaler_clustering = StandardScaler()
        X_scaled = scaler_clustering.fit_transform(X_clustering)
        
        # Entrenar K-Means
        modelo_clustering = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels = modelo_clustering.fit_predict(X_scaled)
        
        # Calcular silhouette score
        clustering_silhouette = silhouette_score(X_scaled, labels)
        
        print(f"✅ Modelo de clustering entrenado. Silhouette: {clustering_silhouette:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Error al entrenar modelo de clustering: {e}")
        return False

def obtener_recomendaciones_prediccion(asistencia: float, prediccion: str, probabilidad: float) -> List[str]:
    """Genera recomendaciones basadas en la predicción"""
    recomendaciones = []
    
    if prediccion == "REPROBADO":
        recomendaciones.append("⚠️ Alto riesgo de reprobación - Se requiere intervención inmediata")
        if asistencia < 75:
            recomendaciones.append("📊 Mejorar asistencia a clases (actual: {:.1f}%)".format(asistencia))
        recomendaciones.append("📚 Solicitar tutorías académicas")
        recomendaciones.append("👥 Formar grupos de estudio")
        recomendaciones.append("⏰ Establecer un horario de estudio estructurado")
    else:
        if probabilidad < 0.7:
            recomendaciones.append("⚠️ Probabilidad moderada - Mantener esfuerzo constante")
        else:
            recomendaciones.append("✅ Buen desempeño - Continuar con el ritmo actual")
        
        if asistencia < 85:
            recomendaciones.append("📊 Mantener o mejorar asistencia (actual: {:.1f}%)".format(asistencia))
        
        recomendaciones.append("📖 Revisar material regularmente")
        recomendaciones.append("🎯 Participar activamente en clases")
    
    return recomendaciones

def obtener_recomendaciones_cluster(cluster: int, asist_prom: float, nota_prom: float) -> tuple:
    """Genera perfil y recomendaciones basadas en el cluster"""
    
    # Determinar perfil según características
    if asist_prom >= 85 and nota_prom >= 7.5:
        perfil = "Estudiante Excelente ⭐"
        descripcion = "Alto rendimiento académico con excelente asistencia y calificaciones."
        recomendaciones = [
            "🌟 Mantener el excelente desempeño",
            "👥 Considerar rol de tutor o mentor",
            "📚 Participar en proyectos avanzados",
            "🎯 Explorar temas de profundización"
        ]
    elif asist_prom >= 70 and nota_prom >= 7.0:
        perfil = "Estudiante Regular 📊"
        descripcion = "Rendimiento promedio con potencial de mejora."
        recomendaciones = [
            "📈 Trabajar en mejorar asistencia y notas",
            "📚 Establecer metas académicas claras",
            "👥 Participar en grupos de estudio",
            "⏰ Optimizar hábitos de estudio"
        ]
    else:
        perfil = "Estudiante en Riesgo ⚠️"
        descripcion = "Bajo rendimiento que requiere atención inmediata."
        recomendaciones = [
            "🚨 Intervención académica urgente",
            "📊 Plan de mejora personalizado",
            "👨‍🏫 Tutorías intensivas",
            "🤝 Apoyo psicopedagógico",
            "📱 Seguimiento semanal del progreso"
        ]
    
    return perfil, descripcion, recomendaciones

# ============================================================================
# ENDPOINTS DE LA API
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Ejecutar al iniciar la API"""
    print("\n" + "="*70)
    print("🚀 INICIANDO API DE ANÁLISIS DE RENDIMIENTO ACADÉMICO")
    print("="*70)
    
    # Cargar datos y entrenar modelos
    if cargar_y_preparar_datos():
        entrenar_modelo_supervisado()
        entrenar_modelo_clustering()
        print("\n✅ API lista para recibir peticiones")
        print("📍 Documentación: http://localhost:8000/docs")
        print("="*70 + "\n")
    else:
        print("\n⚠️ API iniciada con advertencias - Verifica los datos")

@app.get("/", tags=["General"])
async def root():
    """Endpoint raíz - Información de la API"""
    return {
        "message": "API de Análisis de Rendimiento Académico",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/api/v1/predict",
            "cluster": "/api/v1/cluster",
            "stats": "/api/v1/stats"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Verificar estado de salud de la API"""
    return HealthResponse(
        status="healthy" if all([modelo_supervisado, modelo_clustering, df_clean is not None]) else "degraded",
        timestamp=datetime.now().isoformat(),
        modelo_supervisado_cargado=modelo_supervisado is not None,
        modelo_clustering_cargado=modelo_clustering is not None,
        dataset_cargado=df_clean is not None,
        total_registros=len(df_clean) if df_clean is not None else 0
    )

@app.post("/api/v1/predict", response_model=PrediccionResponse, tags=["Modelo Supervisado"])
async def predecir_aprobacion(estudiante: EstudianteInput):
    """
    Predice si un estudiante aprobará o reprobará una asignatura.
    
    **Parámetros:**
    - **asistencia**: Porcentaje de asistencia (0-100)
    - **num_matricula**: Número de matrícula (1, 2, 3...)
    - **nivel**: Nivel académico (PRIMERO, SEGUNDO, TERCERO, CUARTO, etc.)
    
    **Retorna:**
    - Predicción (APROBADO/REPROBADO)
    - Probabilidades
    - Nivel de confianza
    - Recomendaciones personalizadas
    """
    
    if modelo_supervisado is None or scaler_supervisado is None or label_encoder is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible. Verifica que los datos estén cargados.")
    
    try:
        # Codificar nivel
        try:
            nivel_encoded = label_encoder.transform([estudiante.nivel.upper()])[0]
        except:
            # Si el nivel no existe, usar el más común
            nivel_encoded = 0
        
        # Preparar features
        X_input = np.array([[
            estudiante.asistencia,
            estudiante.num_matricula,
            nivel_encoded
        ]])
        
        # Escalar
        X_scaled = scaler_supervisado.transform(X_input)
        
        # Predecir
        prediccion = modelo_supervisado.predict(X_scaled)[0]
        probabilidades = modelo_supervisado.predict_proba(X_scaled)[0]
        
        prob_reprobar = probabilidades[0]
        prob_aprobar = probabilidades[1]
        
        # Determinar confianza
        max_prob = max(prob_aprobar, prob_reprobar)
        if max_prob >= 0.8:
            confianza = "ALTA"
        elif max_prob >= 0.6:
            confianza = "MEDIA"
        else:
            confianza = "BAJA"
        
        # Generar recomendaciones
        prediccion_texto = "APROBADO" if prediccion == 1 else "REPROBADO"
        recomendaciones = obtener_recomendaciones_prediccion(
            estudiante.asistencia,
            prediccion_texto,
            prob_aprobar
        )
        
        return PrediccionResponse(
            prediccion=prediccion_texto,
            probabilidad_aprobar=float(prob_aprobar),
            probabilidad_reprobar=float(prob_reprobar),
            confianza=confianza,
            recomendaciones=recomendaciones,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.post("/api/v1/predict/batch", response_model=List[PrediccionResponse], tags=["Modelo Supervisado"])
async def predecir_aprobacion_batch(estudiantes: List[EstudianteInput]):
    """
    Predice aprobación para múltiples estudiantes.
    
    Útil para procesar listas de estudiantes en lote.
    """
    resultados = []
    
    for i, estudiante in enumerate(estudiantes):
        try:
            resultado = await predecir_aprobacion(estudiante)
            resultado.estudiante_id = f"estudiante_{i+1}"
            resultados.append(resultado)
        except Exception as e:
            # Continuar con los demás aunque uno falle
            print(f"Error procesando estudiante {i+1}: {e}")
    
    return resultados

@app.post("/api/v1/cluster", response_model=ClusterResponse, tags=["Modelo No Supervisado"])
async def asignar_cluster(estudiante: EstudianteClusterInput):
    """
    Asigna un estudiante a un cluster según su rendimiento.
    
    **Parámetros:**
    - **asistencia_promedio**: Asistencia promedio (0-100)
    - **nota_promedio**: Nota promedio (0-10)
    
    **Retorna:**
    - Número de cluster
    - Perfil del estudiante
    - Descripción del perfil
    - Distancia al centroide
    - Recomendaciones personalizadas
    """
    
    if modelo_clustering is None or scaler_clustering is None:
        raise HTTPException(status_code=503, detail="Modelo de clustering no disponible.")
    
    try:
        # Preparar features
        X_input = np.array([[
            estudiante.asistencia_promedio,
            estudiante.nota_promedio
        ]])
        
        # Escalar
        X_scaled = scaler_clustering.transform(X_input)
        
        # Predecir cluster
        cluster = int(modelo_clustering.predict(X_scaled)[0])
        
        # Calcular distancia al centroide
        centroide = modelo_clustering.cluster_centers_[cluster]
        distancia = float(np.linalg.norm(X_scaled[0] - centroide))
        
        # Obtener perfil y recomendaciones
        perfil, descripcion, recomendaciones = obtener_recomendaciones_cluster(
            cluster,
            estudiante.asistencia_promedio,
            estudiante.nota_promedio
        )
        
        return ClusterResponse(
            cluster=cluster,
            perfil=perfil,
            descripcion=descripcion,
            distancia_al_centroide=distancia,
            recomendaciones=recomendaciones
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en clustering: {str(e)}")

@app.get("/api/v1/stats", response_model=EstadisticasResponse, tags=["Estadísticas"])
async def obtener_estadisticas():
    """
    Obtiene estadísticas generales del dataset y los modelos.
    
    Incluye métricas de rendimiento académico y precisión de modelos.
    """
    
    if df_clean is None or student_df is None:
        raise HTTPException(status_code=503, detail="Datos no disponibles.")
    
    return EstadisticasResponse(
        total_registros=len(df_clean),
        total_estudiantes=len(student_df),
        proporcion_aprobados=float(df_clean['Aprobado'].mean()),
        asistencia_promedio=float(student_df['Asistencia_promedio'].mean()),
        nota_promedio=float(student_df['Nota_promedio'].mean()),
        accuracy_modelo=float(modelo_accuracy),
        silhouette_score=float(clustering_silhouette)
    )

@app.get("/api/v1/niveles", tags=["Información"])
async def obtener_niveles_disponibles():
    """Retorna los niveles académicos disponibles en el dataset"""
    
    if df_clean is None:
        raise HTTPException(status_code=503, detail="Datos no disponibles.")
    
    niveles = sorted(df_clean['Nivel'].unique().tolist())
    
    return {
        "niveles": niveles,
        "total": len(niveles)
    }

# ============================================================================
# EJECUTAR LA API
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n🚀 Iniciando servidor API...")
    print("📍 http://localhost:8000")
    print("📚 Documentación: http://localhost:8000/docs\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
