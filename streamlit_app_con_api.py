import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

st.set_page_config(
    page_title="Análisis de Rendimiento Académico - ML con API",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de la API
API_BASE_URL = st.secrets.get("API_URL", "http://localhost:8000")

# ============================================================================
# FUNCIONES DE API
# ============================================================================

def verificar_api():
    """Verifica si la API está disponible"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def predecir_con_api(asistencia, num_matricula, nivel):
    """Hace una predicción usando la API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/predict",
            json={
                "asistencia": float(asistencia),
                "num_matricula": int(num_matricula),
                "nivel": nivel
            },
            timeout=5
        )
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"Error {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def clustering_con_api(asistencia_prom, nota_prom):
    """Hace clustering usando la API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/cluster",
            json={
                "asistencia_promedio": float(asistencia_prom),
                "nota_promedio": float(nota_prom)
            },
            timeout=5
        )
        if response.status_code == 200:
            return True, response.json()
        return False, {"error": f"Error {response.status_code}"}
    except Exception as e:
        return False, {"error": str(e)}

def obtener_estadisticas_api():
    """Obtiene estadísticas desde la API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/stats", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def obtener_niveles_api():
    """Obtiene niveles disponibles desde la API"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/niveles", timeout=5)
        if response.status_code == 200:
            return response.json().get("niveles", ["PRIMERO", "SEGUNDO", "TERCERO", "CUARTO"])
        return ["PRIMERO", "SEGUNDO", "TERCERO", "CUARTO"]
    except:
        return ["PRIMERO", "SEGUNDO", "TERCERO", "CUARTO"]

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

st.title("🎓 Análisis de Rendimiento Académico con ML + API")
st.markdown("**Guía Práctica U3 - Inteligencia Artificial**")
st.markdown("---")

# Verificar estado de la API
api_disponible, health_data = verificar_api()

if api_disponible:
    st.sidebar.success("✅ API Conectada")
    if health_data:
        st.sidebar.info(f"""
        **Estado de la API:**
        - Modelo Supervisado: {'✅' if health_data.get('modelo_supervisado_cargado') else '❌'}
        - Modelo Clustering: {'✅' if health_data.get('modelo_clustering_cargado') else '❌'}
        - Registros: {health_data.get('total_registros', 0):,}
        """)
else:
    st.sidebar.error("❌ API No Disponible")
    st.sidebar.warning("""
    **Para usar esta aplicación:**
    
    1. Abre una nueva terminal
    2. Ejecuta: `python api.py`
    3. Espera a que la API se inicie
    4. Recarga esta página
    """)
    st.stop()

# ============================================================================
# BARRA LATERAL
# ============================================================================

st.sidebar.header("⚙️ Navegación")
pagina = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "🏠 Inicio",
        "🎓 Predicción Individual (API)",
        "📊 Predicción en Lote (API)",
        "🔗 Clustering (API)",
        "📈 Estadísticas (API)",
        "🧪 Prueba la API"
    ]
)

# ============================================================================
# PÁGINA: INICIO
# ============================================================================

if pagina == "🏠 Inicio":
    st.header("🏠 Bienvenido al Sistema de Análisis Académico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 ¿Qué puedes hacer?")
        st.markdown("""
        Esta aplicación integra **Machine Learning** con **APIs REST** para:
        
        1. **Predicción Individual**: Predice si un estudiante aprobará o reprobará
        2. **Predicción en Lote**: Procesa múltiples estudiantes a la vez
        3. **Clustering**: Agrupa estudiantes según su rendimiento
        4. **Estadísticas**: Visualiza métricas generales del dataset
        
        Todas las funcionalidades están respaldadas por una **API REST** desarrollada con FastAPI.
        """)
    
    with col2:
        st.subheader("🚀 Características")
        st.markdown("""
        ✅ **API REST** con FastAPI  
        ✅ **Modelos de ML** entrenados (Random Forest + K-Means)  
        ✅ **Interfaz interactiva** con Streamlit  
        ✅ **Recomendaciones personalizadas**  
        ✅ **Documentación automática** (Swagger)  
        ✅ **Validación de datos** con Pydantic  
        """)
    
    st.markdown("---")
    
    # Obtener estadísticas
    success, stats = obtener_estadisticas_api()
    
    if success:
        st.subheader("📊 Estadísticas del Sistema")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Registros", f"{stats['total_registros']:,}")
        with col2:
            st.metric("👥 Estudiantes", f"{stats['total_estudiantes']:,}")
        with col3:
            st.metric("✅ % Aprobados", f"{stats['proporcion_aprobados']*100:.1f}%")
        with col4:
            st.metric("🎯 Accuracy Modelo", f"{stats['accuracy_modelo']*100:.1f}%")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Asistencia Promedio", f"{stats['asistencia_promedio']:.1f}%")
        with col2:
            st.metric("📝 Nota Promedio", f"{stats['nota_promedio']:.2f}")
        with col3:
            st.metric("🔗 Silhouette Score", f"{stats['silhouette_score']:.4f}")
    
    st.markdown("---")
    
    st.subheader("📚 Documentación de la API")
    st.markdown(f"""
    La API REST está documentada automáticamente. Puedes acceder a:
    
    - **Swagger UI (interactiva)**: [{API_BASE_URL}/docs]({API_BASE_URL}/docs)
    - **ReDoc (documentación)**: [{API_BASE_URL}/redoc]({API_BASE_URL}/redoc)
    
    Desde allí puedes probar todos los endpoints directamente.
    """)

# ============================================================================
# PÁGINA: PREDICCIÓN INDIVIDUAL
# ============================================================================

elif pagina == "🎓 Predicción Individual (API)":
    st.header("🎓 Predicción Individual con API")
    st.markdown("Predice si un estudiante aprobará o reprobará una asignatura")
    
    # Obtener niveles disponibles
    niveles = obtener_niveles_api()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Datos del Estudiante")
        
        asistencia = st.slider(
            "Asistencia (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=0.5,
            help="Porcentaje de asistencia a clases"
        )
        
        num_matricula = st.number_input(
            "Número de Matrícula",
            min_value=1,
            max_value=5,
            value=1,
            help="1 = Primera vez, 2 = Segunda vez, etc."
        )
        
        nivel = st.selectbox(
            "Nivel Académico",
            options=niveles,
            help="Nivel académico del estudiante"
        )
        
        predecir_btn = st.button("🚀 Predecir con API", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Vista Previa de Datos")
        
        datos_preview = pd.DataFrame({
            "Campo": ["Asistencia", "Número de Matrícula", "Nivel"],
            "Valor": [f"{asistencia}%", num_matricula, nivel]
        })
        
        st.dataframe(datos_preview, use_container_width=True, hide_index=True)
        
        st.info("""
        **💡 Consejos:**
        - Asistencia > 80%: Mayor probabilidad de aprobar
        - Primera matrícula (1): Mejor que repetir
        - Los datos se envían a la API en tiempo real
        """)
    
    if predecir_btn:
        with st.spinner("🔄 Consultando API..."):
            success, resultado = predecir_con_api(asistencia, num_matricula, nivel)
        
        if success:
            st.success("✅ Predicción completada")
            
            # Mostrar resultado
            col1, col2, col3 = st.columns(3)
            
            with col1:
                color = "green" if resultado['prediccion'] == "APROBADO" else "red"
                st.markdown(f"### Predicción")
                st.markdown(f"<h2 style='color: {color};'>{resultado['prediccion']}</h2>", unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "Probabilidad de Aprobar",
                    f"{resultado['probabilidad_aprobar']*100:.1f}%",
                    delta=f"{(resultado['probabilidad_aprobar'] - 0.5)*100:.1f}%"
                )
            
            with col3:
                st.metric("Confianza", resultado['confianza'])
            
            # Gráfico de probabilidades
            fig = go.Figure(data=[
                go.Bar(
                    x=['Aprobar', 'Reprobar'],
                    y=[resultado['probabilidad_aprobar']*100, resultado['probabilidad_reprobar']*100],
                    marker_color=['green', 'red'],
                    text=[f"{resultado['probabilidad_aprobar']*100:.1f}%", 
                          f"{resultado['probabilidad_reprobar']*100:.1f}%"],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Probabilidades de Predicción",
                yaxis_title="Probabilidad (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendaciones
            st.subheader("💡 Recomendaciones Personalizadas")
            for i, rec in enumerate(resultado['recomendaciones'], 1):
                st.info(f"{i}. {rec}")
            
            # Timestamp
            st.caption(f"⏰ Predicción realizada: {resultado['timestamp']}")
            
        else:
            st.error(f"❌ Error al consultar la API: {resultado.get('error', 'Error desconocido')}")

# ============================================================================
# PÁGINA: PREDICCIÓN EN LOTE
# ============================================================================

elif pagina == "📊 Predicción en Lote (API)":
    st.header("📊 Predicción en Lote con API")
    st.markdown("Procesa múltiples estudiantes simultáneamente")
    
    st.subheader("📝 Ingresar Datos")
    
    # Opción 1: Entrada manual
    with st.expander("✍️ Entrada Manual de Datos", expanded=True):
        num_estudiantes = st.number_input("Número de estudiantes a predecir", 1, 10, 3)
        
        estudiantes_data = []
        
        for i in range(num_estudiantes):
            st.markdown(f"**Estudiante {i+1}:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                asist = st.number_input(f"Asistencia {i+1} (%)", 0.0, 100.0, 85.0, key=f"asist_{i}")
            with col2:
                mat = st.number_input(f"Matrícula {i+1}", 1, 5, 1, key=f"mat_{i}")
            with col3:
                niv = st.selectbox(f"Nivel {i+1}", ["PRIMERO", "SEGUNDO", "TERCERO"], key=f"niv_{i}")
            
            estudiantes_data.append({
                "asistencia": asist,
                "num_matricula": mat,
                "nivel": niv
            })
    
    if st.button("🚀 Predecir Lote con API", type="primary"):
        with st.spinner("🔄 Procesando lote en la API..."):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/predict/batch",
                    json=estudiantes_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    resultados = response.json()
                    
                    st.success(f"✅ {len(resultados)} estudiantes procesados")
                    
                    # Tabla de resultados
                    df_resultados = pd.DataFrame([
                        {
                            "ID": r.get('estudiante_id', f"Est. {i+1}"),
                            "Predicción": r['prediccion'],
                            "Prob. Aprobar": f"{r['probabilidad_aprobar']*100:.1f}%",
                            "Prob. Reprobar": f"{r['probabilidad_reprobar']*100:.1f}%",
                            "Confianza": r['confianza']
                        }
                        for i, r in enumerate(resultados)
                    ])
                    
                    st.dataframe(df_resultados, use_container_width=True, hide_index=True)
                    
                    # Gráfico de resumen
                    aprobados = sum(1 for r in resultados if r['prediccion'] == 'APROBADO')
                    reprobados = len(resultados) - aprobados
                    
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=['Aprobados', 'Reprobados'],
                            values=[aprobados, reprobados],
                            marker_colors=['green', 'red'],
                            hole=0.4
                        )
                    ])
                    
                    fig.update_layout(title="Resumen de Predicciones", height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"❌ Error en la API: {response.status_code}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ============================================================================
# PÁGINA: CLUSTERING
# ============================================================================

elif pagina == "🔗 Clustering (API)":
    st.header("🔗 Clustering con API")
    st.markdown("Identifica el perfil de un estudiante según su rendimiento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Datos Promedio del Estudiante")
        
        asistencia_prom = st.slider(
            "Asistencia Promedio (%)",
            0.0, 100.0, 80.0, 0.5
        )
        
        nota_prom = st.slider(
            "Nota Promedio",
            0.0, 10.0, 7.5, 0.1
        )
        
        clustering_btn = st.button("🔍 Analizar con API", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("📊 Vista de Datos")
        
        datos = pd.DataFrame({
            "Métrica": ["Asistencia Promedio", "Nota Promedio"],
            "Valor": [f"{asistencia_prom}%", f"{nota_prom:.2f}"]
        })
        
        st.dataframe(datos, use_container_width=True, hide_index=True)
        
        # Visualización de punto
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[asistencia_prom],
            y=[nota_prom],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Tu Perfil'
        ))
        
        fig.update_layout(
            xaxis_title="Asistencia Promedio (%)",
            yaxis_title="Nota Promedio",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    if clustering_btn:
        with st.spinner("🔄 Consultando API..."):
            success, resultado = clustering_con_api(asistencia_prom, nota_prom)
        
        if success:
            st.success("✅ Análisis completado")
            
            # Mostrar resultados
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Cluster Asignado", resultado['cluster'])
            with col2:
                st.metric("Distancia al Centroide", f"{resultado['distancia_al_centroide']:.4f}")
            with col3:
                st.info(resultado['perfil'])
            
            # Descripción del perfil
            st.subheader("📝 Descripción del Perfil")
            st.info(resultado['descripcion'])
            
            # Recomendaciones
            st.subheader("💡 Recomendaciones")
            for i, rec in enumerate(resultado['recomendaciones'], 1):
                if "🚨" in rec or "⚠️" in rec:
                    st.warning(f"{i}. {rec}")
                elif "🌟" in rec or "✅" in rec:
                    st.success(f"{i}. {rec}")
                else:
                    st.info(f"{i}. {rec}")
        else:
            st.error(f"❌ Error: {resultado.get('error', 'Error desconocido')}")

# ============================================================================
# PÁGINA: ESTADÍSTICAS
# ============================================================================

elif pagina == "📈 Estadísticas (API)":
    st.header("📈 Estadísticas Generales (API)")
    
    success, stats = obtener_estadisticas_api()
    
    if success:
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📚 Total Registros", f"{stats['total_registros']:,}")
        with col2:
            st.metric("👥 Estudiantes Únicos", f"{stats['total_estudiantes']:,}")
        with col3:
            st.metric("✅ Proporción Aprobados", f"{stats['proporcion_aprobados']*100:.2f}%")
        with col4:
            st.metric("📊 Asistencia Promedio", f"{stats['asistencia_promedio']:.2f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Rendimiento del Modelo Supervisado")
            st.metric("Accuracy", f"{stats['accuracy_modelo']*100:.2f}%")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stats['accuracy_modelo']*100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Accuracy (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "yellow"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🔗 Calidad del Clustering")
            st.metric("Silhouette Score", f"{stats['silhouette_score']:.4f}")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=stats['silhouette_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Silhouette Score"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 0.3], 'color': "red"},
                        {'range': [0.3, 0.5], 'color': "yellow"},
                        {'range': [0.5, 1], 'color': "green"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("❌ No se pudieron obtener las estadísticas")

# ============================================================================
# PÁGINA: PRUEBA LA API
# ============================================================================

else:  # Prueba la API
    st.header("🧪 Prueba la API Directamente")
    
    st.markdown("""
    Esta sección te permite probar los endpoints de la API directamente desde la interfaz.
    """)
    
    endpoint = st.selectbox(
        "Selecciona un endpoint",
        [
            "/health",
            "/api/v1/stats",
            "/api/v1/niveles",
            "/api/v1/predict (POST)",
            "/api/v1/cluster (POST)"
        ]
    )
    
    if endpoint == "/health":
        if st.button("🔍 Consultar"):
            response = requests.get(f"{API_BASE_URL}/health")
            st.json(response.json())
    
    elif endpoint == "/api/v1/stats":
        if st.button("🔍 Consultar"):
            response = requests.get(f"{API_BASE_URL}/api/v1/stats")
            st.json(response.json())
    
    elif endpoint == "/api/v1/niveles":
        if st.button("🔍 Consultar"):
            response = requests.get(f"{API_BASE_URL}/api/v1/niveles")
            st.json(response.json())
    
    elif endpoint == "/api/v1/predict (POST)":
        st.subheader("Enviar datos JSON")
        
        json_input = st.text_area(
            "JSON de entrada",
            value='{"asistencia": 85.0, "num_matricula": 1, "nivel": "PRIMERO"}',
            height=150
        )
        
        if st.button("📤 Enviar POST"):
            try:
                data = json.loads(json_input)
                response = requests.post(f"{API_BASE_URL}/api/v1/predict", json=data)
                st.json(response.json())
            except Exception as e:
                st.error(f"Error: {e}")
    
    elif endpoint == "/api/v1/cluster (POST)":
        st.subheader("Enviar datos JSON")
        
        json_input = st.text_area(
            "JSON de entrada",
            value='{"asistencia_promedio": 85.0, "nota_promedio": 7.5}',
            height=150
        )
        
        if st.button("📤 Enviar POST"):
            try:
                data = json.loads(json_input)
                response = requests.post(f"{API_BASE_URL}/api/v1/cluster", json=data)
                st.json(response.json())
            except Exception as e:
                st.error(f"Error: {e}")
    
    st.markdown("---")
    st.info(f"""
    💡 **Tip:** También puedes usar la documentación interactiva de Swagger:
    [{API_BASE_URL}/docs]({API_BASE_URL}/docs)
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎓 <b>Guía Práctica U3 - Inteligencia Artificial</b></p>
    <p>Aplicación con integración de API REST | FastAPI + Streamlit</p>
</div>
""", unsafe_allow_html=True)
