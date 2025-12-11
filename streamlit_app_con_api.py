import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    silhouette_score,
    ConfusionMatrixDisplay
)

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Análisis de Rendimiento Académico - ML con API",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# URL de la API
API_BASE_URL = st.secrets.get("API_URL")

# Título principal
st.title("🎓 Análisis de Rendimiento Académico con Machine Learning + API")
st.markdown("**Guía Práctica U3 - Inteligencia Artificial**")
st.markdown("---")

# ============================================================================
# FUNCIONES DE CARGA Y PREPARACIÓN
# ============================================================================
@st.cache_data
def load_data():
    """Carga el dataset desde CSV."""
    try:
        df = pd.read_csv('academic_performance_master.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Error: No se encontró el archivo 'academic_performance_master.csv' en la carpeta MAESTRODENOTAS")
        return None

@st.cache_data
def prepare_data(df):
    """Prepara los datos para los modelos de ML."""
    
    # Seleccionar columnas relevantes
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
    
    # Eliminar nulos
    df_clean.dropna(subset=['Asistencia', 'Nota_final', 'Estado_Asignatura'], inplace=True)
    
    # Corregir valores anómalos
    df_clean['Asistencia'] = df_clean['Asistencia'].clip(lower=0, upper=100)
    df_clean['Nota_final'] = df_clean['Nota_final'].clip(lower=0, upper=10)
    
    # Crear variable objetivo
    df_clean['Aprobado'] = (df_clean['Estado_Asignatura'] == 'APROBADO').astype(int)
    
    # Codificar Nivel
    le_nivel = LabelEncoder()
    df_clean['Nivel_encoded'] = le_nivel.fit_transform(df_clean['Nivel'].astype(str))
    
    # Agregar por estudiante para clustering
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
    
    return df_clean, student_df, le_nivel

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

# ============================================================================
# BARRA LATERAL
# ============================================================================
st.sidebar.header("⚙️ Configuración")

# Cargar datos
df = load_data()

if df is None:
    st.stop()

df_clean, student_df, le_nivel = prepare_data(df)

st.sidebar.success(f"✅ Datos cargados: {len(df)} registros")
st.sidebar.info(f"📊 Estudiantes únicos: {len(student_df)}")

# Verificar API
api_disponible, health_data = verificar_api()
if api_disponible:
    st.sidebar.success("✅ API Conectada")
else:
    st.sidebar.info("ℹ️ API no disponible (modo local)")

# Selector de página
st.sidebar.markdown("---")
pagina = st.sidebar.radio(
    "Selecciona una sección:",
    [
        "📊 Exploración de Datos",
        "🎓 Modelo Supervisado (Clasificación)",
        "🔗 Modelo No Supervisado (Clustering)",
        "📈 Comparación de Modelos"
    ]
)

# ============================================================================
# PÁGINA 1: EXPLORACIÓN DE DATOS
# ============================================================================
if pagina == "📊 Exploración de Datos":
    st.header("📊 Exploración de Datos")
    
    tab1, tab2, tab3 = st.tabs(["Vista General", "Distribuciones", "Relaciones"])
    
    with tab1:
        st.subheader("🔍 Vista General del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Registros", f"{len(df):,}")
        with col2:
            st.metric("Estudiantes Únicos", f"{df['Identificacion_Estudiante'].nunique():,}")
        with col3:
            st.metric("Asignaturas Únicas", f"{df['Asignatura'].nunique():,}")
        with col4:
            aprobados = df[df['Estado_Asignatura'] == 'APROBADO'].shape[0]
            prop_aprobados = (aprobados / len(df)) * 100
            st.metric("% Aprobados", f"{prop_aprobados:.1f}%")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Primeras Filas")
            st.dataframe(df_clean.head(10), use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("📈 Estadísticas Descriptivas")
            st.dataframe(df_clean[['Asistencia', 'Nota_final', 'Num_matricula']].describe(),
                        use_container_width=True)
    
    with tab2:
        st.subheader("📊 Distribuciones de Variables")
        
        # Estado de Asignatura
        estado_counts = df_clean['Estado_Asignatura'].value_counts()
        fig_estado = px.bar(
            x=estado_counts.index,
            y=estado_counts.values,
            labels={'x': 'Estado', 'y': 'Cantidad'},
            title='Distribución de Estado de Asignatura',
            color=estado_counts.index,
            color_discrete_map={'APROBADO': '#2ecc71', 'REPROBADO': '#e74c3c'}
        )
        fig_estado.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_estado, use_container_width=True)
        
        # Distribuciones numéricas
        col1, col2 = st.columns(2)
        
        with col1:
            fig_asist = px.histogram(
                df_clean,
                x='Asistencia',
                nbins=50,
                title='Distribución de Asistencia',
                labels={'Asistencia': 'Asistencia (%)'},
                color_discrete_sequence=['#3498db']
            )
            fig_asist.add_vline(
                x=df_clean['Asistencia'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Media: {df_clean['Asistencia'].mean():.1f}%"
            )
            st.plotly_chart(fig_asist, use_container_width=True)
        
        with col2:
            fig_nota = px.histogram(
                df_clean,
                x='Nota_final',
                nbins=50,
                title='Distribución de Nota Final',
                labels={'Nota_final': 'Nota Final'},
                color_discrete_sequence=['#9b59b6']
            )
            fig_nota.add_vline(
                x=df_clean['Nota_final'].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Media: {df_clean['Nota_final'].mean():.2f}"
            )
            st.plotly_chart(fig_nota, use_container_width=True)
    
    with tab3:
        st.subheader("🔗 Relación entre Variables")
        
        # Scatter plot: Asistencia vs Nota Final
        fig_scatter = px.scatter(
            df_clean.sample(min(5000, len(df_clean))),  # Muestra para performance
            x='Asistencia',
            y='Nota_final',
            color='Aprobado',
            title='Relación entre Asistencia y Nota Final',
            labels={'Asistencia': 'Asistencia (%)', 'Nota_final': 'Nota Final', 'Aprobado': 'Estado'},
            color_discrete_map={0: '#e74c3c', 1: '#2ecc71'},
            opacity=0.5
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlación
        corr = df_clean[['Asistencia', 'Nota_final']].corr().iloc[0, 1]
        st.info(f"📊 **Correlación entre Asistencia y Nota Final:** {corr:.4f}")
        
        # Box plots
        col1, col2 = st.columns(2)
        
        with col1:
            fig_box1 = px.box(
                df_clean,
                x='Aprobado',
                y='Asistencia',
                title='Asistencia por Estado',
                labels={'Aprobado': 'Estado (0=Reprobado, 1=Aprobado)', 'Asistencia': 'Asistencia (%)'},
                color='Aprobado',
                color_discrete_map={0: '#e74c3c', 1: '#2ecc71'}
            )
            st.plotly_chart(fig_box1, use_container_width=True)
        
        with col2:
            fig_box2 = px.box(
                df_clean,
                x='Aprobado',
                y='Nota_final',
                title='Nota Final por Estado',
                labels={'Aprobado': 'Estado (0=Reprobado, 1=Aprobado)', 'Nota_final': 'Nota Final'},
                color='Aprobado',
                color_discrete_map={0: '#e74c3c', 1: '#2ecc71'}
            )
            st.plotly_chart(fig_box2, use_container_width=True)

# ============================================================================
# PÁGINA 2: MODELO SUPERVISADO
# ============================================================================
elif pagina == "🎓 Modelo Supervisado (Clasificación)":
    st.header("🎓 Modelo Supervisado: Clasificación")
    st.markdown("**Objetivo:** Predecir si un estudiante aprobará o reprobará una asignatura")
    
    # Parámetros en sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Parámetros del Modelo")
    
    modelo_tipo = st.sidebar.selectbox(
        "Selecciona el algoritmo:",
        ["Random Forest", "Regresión Logística", "Árbol de Decisión"]
    )
    
    test_size = st.sidebar.slider("Tamaño del conjunto de prueba (%)", 10, 40, 25) / 100
    
    if modelo_tipo == "Random Forest":
        n_estimators = st.sidebar.slider("Número de árboles", 50, 200, 100, 10)
        max_depth = st.sidebar.slider("Profundidad máxima", 5, 30, 15)
    elif modelo_tipo == "Árbol de Decisión":
        max_depth = st.sidebar.slider("Profundidad máxima", 3, 20, 10)
    
    entrenar = st.sidebar.button("🚀 Entrenar Modelo", type="primary")
    
    if entrenar or 'modelo_entrenado' in st.session_state:
        
        # Preparar datos
        feature_columns = ['Asistencia', 'Num_matricula', 'Nivel_encoded']
        X = df_clean[feature_columns].values
        y = df_clean['Aprobado'].values
        
        # Escalar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Dividir
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Entrenar modelo
        if modelo_tipo == "Random Forest":
            modelo = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
        elif modelo_tipo == "Regresión Logística":
            modelo = LogisticRegression(random_state=42, max_iter=1000)
        else:  # Árbol de Decisión
            modelo = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        
        with st.spinner("Entrenando modelo..."):
            modelo.fit(X_train, y_train)
        
        # Predicciones
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.session_state['modelo_entrenado'] = True
        
        # Mostrar resultados
        st.success(f"✅ Modelo entrenado exitosamente")
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        
        with col1:
            st.metric("🎯 Accuracy", f"{accuracy*100:.2f}%")
        with col2:
            st.metric("🎓 Precisión (Aprobados)", f"{report_dict['1']['precision']*100:.2f}%")
        with col3:
            st.metric("📊 Recall (Aprobados)", f"{report_dict['1']['recall']*100:.2f}%")
        with col4:
            st.metric("⚖️ F1-Score (Aprobados)", f"{report_dict['1']['f1-score']*100:.2f}%")
        
        st.markdown("---")
        
        # Tabs para diferentes análisis
        tab1, tab2, tab3 = st.tabs(["Matriz de Confusión", "Reporte Detallado", "Importancia de Características"])
        
        with tab1:
            st.subheader("📊 Matriz de Confusión")
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Reprobado', 'Aprobado'],
                y=['Reprobado', 'Aprobado'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 20},
                colorscale='Blues',
                showscale=True
            ))
            
            fig_cm.update_layout(
                title=f'Matriz de Confusión - {modelo_tipo}',
                xaxis_title='Predicción',
                yaxis_title='Real',
                height=500
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
            
            # Interpretación
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"""
                **Interpretación:**
                - **Verdaderos Negativos (TN):** {cm[0,0]} - Correctamente predichos como Reprobados
                - **Falsos Positivos (FP):** {cm[0,1]} - Incorrectamente predichos como Aprobados
                """)
            with col2:
                st.info(f"""
                \n
                - **Falsos Negativos (FN):** {cm[1,0]} - Incorrectamente predichos como Reprobados
                - **Verdaderos Positivos (TP):** {cm[1,1]} - Correctamente predichos como Aprobados
                """)
        
        with tab2:
            st.subheader("📋 Reporte de Clasificación")
            
            report_df = pd.DataFrame(report_dict).transpose()
            report_df = report_df.round(4)
            
            st.dataframe(
                report_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']),
                use_container_width=True
            )
            
            st.markdown("""
            **Definiciones:**
            - **Precision:** Proporción de predicciones positivas correctas
            - **Recall:** Proporción de positivos reales correctamente identificados
            - **F1-Score:** Media armónica entre Precision y Recall
            - **Support:** Número de muestras de cada clase
            """)
        
        with tab3:
            st.subheader("🔍 Importancia de Características")
            
            if modelo_tipo in ["Random Forest", "Árbol de Decisión"]:
                feature_importance = pd.DataFrame({
                    'Característica': feature_columns,
                    'Importancia': modelo.feature_importances_
                }).sort_values('Importancia', ascending=False)
                
                fig_importance = px.bar(
                    feature_importance,
                    x='Importancia',
                    y='Característica',
                    orientation='h',
                    title=f'Importancia de Características - {modelo_tipo}',
                    color='Importancia',
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(fig_importance, use_container_width=True)
                
                st.success(f"🎯 **Característica más importante:** {feature_importance.iloc[0]['Característica']}")
                st.info(f"💡 La **{feature_importance.iloc[0]['Característica']}** es el factor más influyente para predecir si un estudiante aprobará.")
            else:
                st.info("La importancia de características solo está disponible para modelos basados en árboles.")
    
    else:
        st.info("👈 Configura los parámetros en la barra lateral y presiona **'Entrenar Modelo'** para comenzar.")
        
        st.markdown("""
        ### 📚 Sobre el Modelo Supervisado
        
        Este modelo utiliza **aprendizaje supervisado** para predecir si un estudiante aprobará o reprobará una asignatura.
        
        **Características utilizadas:**
        - 📊 **Asistencia:** Porcentaje de asistencia del estudiante
        - 🔢 **Número de Matrícula:** Indica si es la primera vez que toma la materia
        - 📚 **Nivel:** Nivel académico codificado
        
        **Objetivo:**
        - Predecir la clase: **Aprobado (1)** o **Reprobado (0)**
        
        **Algoritmos disponibles:**
        - **Random Forest:** Ensemble de árboles de decisión
        - **Regresión Logística:** Modelo lineal probabilístico
        - **Árbol de Decisión:** Modelo basado en reglas
        """)

# ============================================================================
# PÁGINA 3: MODELO NO SUPERVISADO
# ============================================================================
elif pagina == "🔗 Modelo No Supervisado (Clustering)":
    st.header("🔗 Modelo No Supervisado: Clustering")
    st.markdown("**Objetivo:** Agrupar estudiantes según patrones en su rendimiento académico")
    
    # Parámetros en sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Parámetros del Clustering")
    
    n_clusters = st.sidebar.slider("Número de Clusters (K)", 2, 6, 3)
    
    ejecutar_clustering = st.sidebar.button("🚀 Ejecutar Clustering", type="primary")
    
    if ejecutar_clustering or 'clustering_ejecutado' in st.session_state:
        
        # Preparar datos
        feature_columns = ['Asistencia_promedio', 'Nota_promedio']
        X_clustering = student_df[feature_columns].values
        
        # Escalar
        scaler_clustering = StandardScaler()
        X_clustering_scaled = scaler_clustering.fit_transform(X_clustering)
        
        # Entrenar K-Means
        with st.spinner(f"Ejecutando K-Means con {n_clusters} clusters..."):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_clustering_scaled)
        
        silhouette = silhouette_score(X_clustering_scaled, labels)
        
        st.session_state['clustering_ejecutado'] = True
        
        # Agregar labels al DataFrame
        student_df['Cluster'] = labels
        
        st.success(f"✅ Clustering completado con K={n_clusters}")
        
        # Métricas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📊 Número de Clusters", n_clusters)
        with col2:
            st.metric("🎯 Silhouette Score", f"{silhouette:.4f}")
        with col3:
            st.metric("👥 Estudiantes Agrupados", len(student_df))
        
        st.markdown("---")
        
        # Tabs
        tab1, tab2, tab3 = st.tabs(["Visualización", "Estadísticas por Cluster", "Método del Codo"])
        
        with tab1:
            st.subheader("📊 Visualización de Clusters")
            
            # Scatter plot con Plotly
            fig_clusters = px.scatter(
                student_df,
                x='Asistencia_promedio',
                y='Nota_promedio',
                color='Cluster',
                title=f'K-Means Clustering (K={n_clusters})',
                labels={
                    'Asistencia_promedio': 'Asistencia Promedio (%)',
                    'Nota_promedio': 'Nota Promedio',
                    'Cluster': 'Cluster'
                },
                color_continuous_scale='Viridis',
                hover_data=['Estudiante']
            )
            
            # Agregar centroides
            centroids_original = scaler_clustering.inverse_transform(kmeans.cluster_centers_)
            
            fig_clusters.add_trace(go.Scatter(
                x=centroids_original[:, 0],
                y=centroids_original[:, 1],
                mode='markers',
                marker=dict(
                    size=20,
                    color='red',
                    symbol='x',
                    line=dict(width=2, color='black')
                ),
                name='Centroides',
                showlegend=True
            ))
            
            fig_clusters.update_layout(height=600)
            st.plotly_chart(fig_clusters, use_container_width=True)
            
            # Distribución de clusters
            col1, col2 = st.columns(2)
            
            with col1:
                cluster_counts = student_df['Cluster'].value_counts().sort_index()
                fig_dist = px.bar(
                    x=cluster_counts.index,
                    y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Cantidad de Estudiantes'},
                    title='Distribución de Estudiantes por Cluster',
                    color=cluster_counts.index,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                # Pie chart
                fig_pie = px.pie(
                    values=cluster_counts.values,
                    names=cluster_counts.index,
                    title='Proporción de Estudiantes por Cluster'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab2:
            st.subheader("📊 Estadísticas por Cluster")
            
            for cluster in range(n_clusters):
                cluster_data = student_df[student_df['Cluster'] == cluster]
                asist_mean = cluster_data['Asistencia_promedio'].mean()
                nota_mean = cluster_data['Nota_promedio'].mean()
                
                # Determinar perfil
                if asist_mean >= 85 and nota_mean >= 7.5:
                    perfil = "Excelentes ⭐"
                    color = "green"
                elif asist_mean >= 70 and nota_mean >= 7.0:
                    perfil = "Regulares 📊"
                    color = "blue"
                else:
                    perfil = "En Riesgo ⚠️"
                    color = "red"
                
                with st.expander(f"🔵 **Cluster {cluster}** - {perfil}", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("👥 Estudiantes", len(cluster_data))
                    with col2:
                        st.metric("📊 Asistencia Promedio", f"{asist_mean:.1f}%")
                    with col3:
                        st.metric("📝 Nota Promedio", f"{nota_mean:.2f}")
                    
                    # Box plots por cluster
                    fig_box = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=('Distribución de Asistencia', 'Distribución de Notas')
                    )
                    
                    fig_box.add_trace(
                        go.Box(y=cluster_data['Asistencia_promedio'], name='Asistencia'),
                        row=1, col=1
                    )
                    
                    fig_box.add_trace(
                        go.Box(y=cluster_data['Nota_promedio'], name='Nota'),
                        row=1, col=2
                    )
                    
                    fig_box.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig_box, use_container_width=True)
                    
                    # Interpretación
                    if perfil == "Excelentes ⭐":
                        st.success("""
                        **Interpretación:** Estudiantes de alto rendimiento con excelente asistencia y calificaciones.
                        Son el modelo a seguir y pueden servir como tutores o mentores.
                        """)
                    elif perfil == "Regulares 📊":
                        st.info("""
                        **Interpretación:** Estudiantes con rendimiento promedio.
                        Tienen potencial de mejora con apoyo adicional y estrategias de estudio.
                        """)
                    else:
                        st.warning("""
                        **Interpretación:** Estudiantes en riesgo académico.
                        Requieren intervención inmediata, tutorías y seguimiento personalizado.
                        """)
        
        with tab3:
            st.subheader("📉 Método del Codo y Silhouette Score")
            
            # Calcular para diferentes K
            inertias = []
            silhouette_scores = []
            K_range = range(2, 11)
            
            with st.spinner("Calculando métricas para diferentes valores de K..."):
                for k in K_range:
                    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels_temp = kmeans_temp.fit_predict(X_clustering_scaled)
                    inertias.append(kmeans_temp.inertia_)
                    silhouette_scores.append(silhouette_score(X_clustering_scaled, labels_temp))
            
            # Gráficos
            fig_metrics = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Método del Codo', 'Silhouette Score')
            )
            
            fig_metrics.add_trace(
                go.Scatter(x=list(K_range), y=inertias, mode='lines+markers', name='Inercia'),
                row=1, col=1
            )
            
            fig_metrics.add_trace(
                go.Scatter(x=list(K_range), y=silhouette_scores, mode='lines+markers', 
                          name='Silhouette', line=dict(color='red')),
                row=1, col=2
            )
            
            # Marcar K actual
            fig_metrics.add_vline(x=n_clusters, line_dash="dash", line_color="green", 
                                 annotation_text=f"K={n_clusters}", row=1, col=1)
            fig_metrics.add_vline(x=n_clusters, line_dash="dash", line_color="green",
                                 annotation_text=f"K={n_clusters}", row=1, col=2)
            
            fig_metrics.update_xaxes(title_text="Número de Clusters (K)")
            fig_metrics.update_yaxes(title_text="Inercia", row=1, col=1)
            fig_metrics.update_yaxes(title_text="Silhouette Score", row=1, col=2)
            
            fig_metrics.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            best_k_silhouette = K_range[np.argmax(silhouette_scores)]
            st.info(f"💡 **K óptimo según Silhouette Score:** K = {best_k_silhouette}")
    
    else:
        st.info("👈 Configura el número de clusters en la barra lateral y presiona **'Ejecutar Clustering'**.")
        
        st.markdown("""
        ### 📚 Sobre el Clustering
        
        Este modelo utiliza **aprendizaje no supervisado** para identificar grupos naturales de estudiantes.
        
        **Características utilizadas:**
        - 📊 **Asistencia Promedio:** Promedio de asistencia del estudiante
        - 📝 **Nota Promedio:** Promedio de calificaciones del estudiante
        
        **Algoritmo:** K-Means
        
        **Objetivo:**
        - Descubrir patrones ocultos en el rendimiento estudiantil
        - Segmentar estudiantes para personalizar estrategias educativas
        - Identificar grupos de riesgo que requieren atención
        
        **Métricas de evaluación:**
        - **Silhouette Score:** Mide qué tan bien definidos están los clusters (valores cercanos a 1 son mejores)
        - **Método del Codo:** Ayuda a encontrar el número óptimo de clusters
        """)

# ============================================================================
# PÁGINA 4: COMPARACIÓN DE MODELOS
# ============================================================================
else:  # Comparación de Modelos
    st.header("📈 Comparación de Modelos")
    st.markdown("**Análisis comparativo entre modelos supervisados y no supervisados**")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎓 Modelo Supervisado")
        st.markdown("""
        **Tipo:** Aprendizaje Supervisado  
        **Algoritmo:** Random Forest / Regresión Logística / Árbol de Decisión
        
        **Objetivo:**
        - Predecir si un estudiante **aprobará** o **reprobará**
        
        **Entrada:**
        - Características etiquetadas (Asistencia, Num_matricula, Nivel)
        
        **Salida:**
        - Clase: **Aprobado (1)** / **Reprobado (0)**
        
        **Métrica principal:**
        - **Accuracy:** Porcentaje de predicciones correctas
        
        **Ventajas:**
        - ✅ Predicciones específicas y accionables
        - ✅ Permite identificar factores influyentes
        - ✅ Útil para intervenciones tempranas
        
        **Aplicaciones:**
        - 🎯 Predecir estudiantes en riesgo
        - 📊 Alertas tempranas de reprobación
        - 🔍 Identificar factores críticos de éxito
        """)
    
    with col2:
        st.subheader("🔗 Modelo No Supervisado")
        st.markdown("""
        **Tipo:** Aprendizaje No Supervisado  
        **Algoritmo:** K-Means Clustering
        
        **Objetivo:**
        - Agrupar estudiantes con **patrones similares**
        
        **Entrada:**
        - Características sin etiquetar (Asistencia_promedio, Nota_promedio)
        
        **Salida:**
        - Etiqueta de cluster: **0, 1, 2, ..., K-1**
        
        **Métrica principal:**
        - **Silhouette Score:** Calidad de los clusters
        
        **Ventajas:**
        - ✅ Descubre patrones ocultos
        - ✅ No requiere etiquetas previas
        - ✅ Segmentación natural de estudiantes
        
        **Aplicaciones:**
        - 📊 Segmentación de estudiantes
        - 🎯 Personalización de estrategias educativas
        - 🔍 Identificación de perfiles académicos
        """)
    
    st.markdown("---")
    
    st.subheader("🏆 ¿Cuál modelo es mejor?")
    
    st.info("""
    **Respuesta:** ¡Depende del objetivo!
    
    Ambos modelos son **complementarios** y resuelven problemas diferentes:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Usa el Modelo Supervisado cuando:**
        - ✅ Necesites **predicciones específicas** sobre un estudiante
        - ✅ Quieras identificar **qué factores** influyen más
        - ✅ Necesites tomar **decisiones individuales**
        - ✅ Tengas datos etiquetados disponibles
        
        **Ejemplo:**  
        "¿Aprobará Juan la asignatura de Matemáticas?"
        """)
    
    with col2:
        st.success("""
        **Usa el Modelo No Supervisado cuando:**
        - ✅ Quieras entender **patrones generales**
        - ✅ Necesites **segmentar** grupos de estudiantes
        - ✅ No tengas etiquetas disponibles
        - ✅ Busques **insights exploratorios**
        
        **Ejemplo:**  
        "¿Qué perfiles de estudiantes existen en mi institución?"
        """)
    
    st.markdown("---")
    
    st.subheader("💡 Recomendación: Estrategia Combinada")
    
    st.success("""
    **Mejor enfoque:** Usar **ambos modelos** de forma complementaria:
    
    1. **Paso 1 - Clustering (No Supervisado):**
       - Identificar grupos naturales de estudiantes
       - Entender perfiles académicos generales
       - Ejemplo: "Estudiantes Excelentes", "Regulares", "En Riesgo"
    
    2. **Paso 2 - Clasificación (Supervisado):**
       - Predecir el desempeño individual dentro de cada grupo
       - Identificar factores específicos de riesgo
       - Tomar acciones preventivas personalizadas
    
    **Beneficios:**
    - 🎯 Estrategias personalizadas por grupo (clustering)
    - 📊 Predicciones precisas a nivel individual (clasificación)
    - 🔍 Intervenciones tempranas y efectivas
    - 💡 Mejor comprensión global del rendimiento estudiantil
    """)
    
    st.markdown("---")
    
    # Tabla comparativa
    st.subheader("📊 Tabla Comparativa")
    
    comparacion_df = pd.DataFrame({
        'Aspecto': [
            'Tipo de Aprendizaje',
            'Requiere Etiquetas',
            'Objetivo Principal',
            'Salida',
            'Métrica Principal',
            'Interpretabilidad',
            'Uso Principal'
        ],
        'Modelo Supervisado': [
            'Supervisado',
            'Sí',
            'Predicción de clase',
            'Aprobado / Reprobado',
            'Accuracy, Precision, Recall',
            'Alta (importancia de características)',
            'Predicciones individuales'
        ],
        'Modelo No Supervisado': [
            'No Supervisado',
            'No',
            'Agrupamiento',
            'Número de cluster',
            'Silhouette Score',
            'Media (requiere análisis)',
            'Segmentación y exploración'
        ]
    })
    
    st.dataframe(comparacion_df, use_container_width=True, hide_index=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(f"""
<div style='text-align: center'>
    <p>🎓 <b>Guía Práctica U3 - Inteligencia Artificial</b></p>
    <p>Versión con integración de API REST | Datos: academic_performance_master.csv</p>
    <p><small>API: {API_BASE_URL}</small></p>
</div>
""", unsafe_allow_html=True)
