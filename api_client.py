import requests
import json
from typing import Dict, List

# Configuración
API_BASE_URL = "http://localhost:8000"

class APIClient:
    """Cliente para interactuar con la API de Rendimiento Académico"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Verifica el estado de la API"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predecir_aprobacion(
        self,
        asistencia: float,
        num_matricula: int,
        nivel: str
    ) -> Dict:
        """
        Predice si un estudiante aprobará o reprobará.
        
        Args:
            asistencia: Porcentaje de asistencia (0-100)
            num_matricula: Número de matrícula
            nivel: Nivel académico (PRIMERO, SEGUNDO, etc.)
        
        Returns:
            Diccionario con la predicción y recomendaciones
        """
        endpoint = f"{self.base_url}/api/v1/predict"
        
        data = {
            "asistencia": asistencia,
            "num_matricula": num_matricula,
            "nivel": nivel
        }
        
        try:
            response = self.session.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def predecir_batch(self, estudiantes: List[Dict]) -> List[Dict]:
        """
        Predice aprobación para múltiples estudiantes.
        
        Args:
            estudiantes: Lista de diccionarios con datos de estudiantes
        
        Returns:
            Lista de predicciones
        """
        endpoint = f"{self.base_url}/api/v1/predict/batch"
        
        try:
            response = self.session.post(endpoint, json=estudiantes)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return [{"error": str(e)}]
    
    def asignar_cluster(
        self,
        asistencia_promedio: float,
        nota_promedio: float
    ) -> Dict:
        """
        Asigna un estudiante a un cluster.
        
        Args:
            asistencia_promedio: Asistencia promedio (0-100)
            nota_promedio: Nota promedio (0-10)
        
        Returns:
            Diccionario con el cluster y recomendaciones
        """
        endpoint = f"{self.base_url}/api/v1/cluster"
        
        data = {
            "asistencia_promedio": asistencia_promedio,
            "nota_promedio": nota_promedio
        }
        
        try:
            response = self.session.post(endpoint, json=data)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def obtener_estadisticas(self) -> Dict:
        """Obtiene estadísticas generales del dataset"""
        endpoint = f"{self.base_url}/api/v1/stats"
        
        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def obtener_niveles(self) -> Dict:
        """Obtiene los niveles académicos disponibles"""
        endpoint = f"{self.base_url}/api/v1/niveles"
        
        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}


def mostrar_resultado_prediccion(resultado: Dict):
    """Imprime el resultado de una predicción de forma legible"""
    print("\n" + "="*70)
    print("📊 RESULTADO DE PREDICCIÓN")
    print("="*70)
    
    if "error" in resultado:
        print(f"❌ Error: {resultado['error']}")
        return
    
    print(f"\n🎯 Predicción: {resultado['prediccion']}")
    print(f"📈 Probabilidad de Aprobar: {resultado['probabilidad_aprobar']*100:.2f}%")
    print(f"📉 Probabilidad de Reprobar: {resultado['probabilidad_reprobar']*100:.2f}%")
    print(f"⚖️ Confianza: {resultado['confianza']}")
    
    print(f"\n💡 Recomendaciones:")
    for i, rec in enumerate(resultado['recomendaciones'], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n⏰ Timestamp: {resultado['timestamp']}")
    print("="*70)


def mostrar_resultado_cluster(resultado: Dict):
    """Imprime el resultado de clustering de forma legible"""
    print("\n" + "="*70)
    print("🔗 RESULTADO DE CLUSTERING")
    print("="*70)
    
    if "error" in resultado:
        print(f"❌ Error: {resultado['error']}")
        return
    
    print(f"\n📊 Cluster: {resultado['cluster']}")
    print(f"👤 Perfil: {resultado['perfil']}")
    print(f"📝 Descripción: {resultado['descripcion']}")
    print(f"📏 Distancia al Centroide: {resultado['distancia_al_centroide']:.4f}")
    
    print(f"\n💡 Recomendaciones:")
    for i, rec in enumerate(resultado['recomendaciones'], 1):
        print(f"   {i}. {rec}")
    
    print("="*70)


def ejemplo_prediccion_individual():
    """Ejemplo 1: Predicción individual"""
    print("\n" + "="*70)
    print("📚 EJEMPLO 1: PREDICCIÓN INDIVIDUAL")
    print("="*70)
    
    client = APIClient()
    
    # Ejemplo: Estudiante con buena asistencia
    print("\n🎓 Caso 1: Estudiante con alta asistencia")
    resultado = client.predecir_aprobacion(
        asistencia=92.5,
        num_matricula=1,
        nivel="PRIMERO"
    )
    mostrar_resultado_prediccion(resultado)
    
    # Ejemplo: Estudiante con baja asistencia
    print("\n🎓 Caso 2: Estudiante con baja asistencia")
    resultado = client.predecir_aprobacion(
        asistencia=55.0,
        num_matricula=2,
        nivel="SEGUNDO"
    )
    mostrar_resultado_prediccion(resultado)


def ejemplo_prediccion_batch():
    """Ejemplo 2: Predicción en lote"""
    print("\n" + "="*70)
    print("📚 EJEMPLO 2: PREDICCIÓN EN LOTE")
    print("="*70)
    
    client = APIClient()
    
    estudiantes = [
        {"asistencia": 85.0, "num_matricula": 1, "nivel": "PRIMERO"},
        {"asistencia": 70.5, "num_matricula": 1, "nivel": "SEGUNDO"},
        {"asistencia": 95.0, "num_matricula": 1, "nivel": "TERCERO"},
        {"asistencia": 60.0, "num_matricula": 2, "nivel": "PRIMERO"},
    ]
    
    resultados = client.predecir_batch(estudiantes)
    
    for i, resultado in enumerate(resultados, 1):
        print(f"\n--- Estudiante {i} ---")
        if "error" not in resultado:
            print(f"Predicción: {resultado['prediccion']}")
            print(f"Probabilidad Aprobar: {resultado['probabilidad_aprobar']*100:.2f}%")
            print(f"Confianza: {resultado['confianza']}")
        else:
            print(f"Error: {resultado['error']}")


def ejemplo_clustering():
    """Ejemplo 3: Clustering"""
    print("\n" + "="*70)
    print("📚 EJEMPLO 3: CLUSTERING")
    print("="*70)
    
    client = APIClient()
    
    # Estudiante excelente
    print("\n🌟 Caso 1: Estudiante excelente")
    resultado = client.asignar_cluster(
        asistencia_promedio=95.0,
        nota_promedio=9.0
    )
    mostrar_resultado_cluster(resultado)
    
    # Estudiante regular
    print("\n📊 Caso 2: Estudiante regular")
    resultado = client.asignar_cluster(
        asistencia_promedio=75.0,
        nota_promedio=7.2
    )
    mostrar_resultado_cluster(resultado)
    
    # Estudiante en riesgo
    print("\n⚠️ Caso 3: Estudiante en riesgo")
    resultado = client.asignar_cluster(
        asistencia_promedio=55.0,
        nota_promedio=5.5
    )
    mostrar_resultado_cluster(resultado)


def ejemplo_estadisticas():
    """Ejemplo 4: Obtener estadísticas"""
    print("\n" + "="*70)
    print("📚 EJEMPLO 4: ESTADÍSTICAS GENERALES")
    print("="*70)
    
    client = APIClient()
    
    stats = client.obtener_estadisticas()
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return
    
    print(f"\n📊 Estadísticas del Dataset:")
    print(f"   - Total de registros: {stats['total_registros']:,}")
    print(f"   - Total de estudiantes: {stats['total_estudiantes']:,}")
    print(f"   - Proporción de aprobados: {stats['proporcion_aprobados']*100:.2f}%")
    print(f"   - Asistencia promedio: {stats['asistencia_promedio']:.2f}%")
    print(f"   - Nota promedio: {stats['nota_promedio']:.2f}")
    
    print(f"\n🎯 Rendimiento de Modelos:")
    print(f"   - Accuracy (Modelo Supervisado): {stats['accuracy_modelo']*100:.2f}%")
    print(f"   - Silhouette Score (Clustering): {stats['silhouette_score']:.4f}")


def ejemplo_health_check():
    """Ejemplo 5: Verificar estado de la API"""
    print("\n" + "="*70)
    print("📚 EJEMPLO 5: HEALTH CHECK")
    print("="*70)
    
    client = APIClient()
    
    health = client.health_check()
    
    if "error" in health:
        print(f"❌ API no disponible: {health['error']}")
        return
    
    print(f"\n✅ Estado de la API: {health['status']}")
    print(f"⏰ Timestamp: {health['timestamp']}")
    print(f"\n📊 Estado de Componentes:")
    print(f"   - Modelo Supervisado: {'✅ Cargado' if health['modelo_supervisado_cargado'] else '❌ No cargado'}")
    print(f"   - Modelo Clustering: {'✅ Cargado' if health['modelo_clustering_cargado'] else '❌ No cargado'}")
    print(f"   - Dataset: {'✅ Cargado' if health['dataset_cargado'] else '❌ No cargado'}")
    print(f"   - Total registros: {health['total_registros']:,}")


def menu_interactivo():
    """Menú interactivo para probar la API"""
    client = APIClient()
    
    while True:
        print("\n" + "="*70)
        print("🎓 CLIENTE API - ANÁLISIS DE RENDIMIENTO ACADÉMICO")
        print("="*70)
        print("\nSelecciona una opción:")
        print("1. Health Check")
        print("2. Predicción Individual")
        print("3. Predicción en Lote")
        print("4. Clustering")
        print("5. Estadísticas Generales")
        print("6. Ver Niveles Disponibles")
        print("7. Ejecutar Todos los Ejemplos")
        print("0. Salir")
        
        opcion = input("\n👉 Opción: ").strip()
        
        if opcion == "1":
            ejemplo_health_check()
        elif opcion == "2":
            ejemplo_prediccion_individual()
        elif opcion == "3":
            ejemplo_prediccion_batch()
        elif opcion == "4":
            ejemplo_clustering()
        elif opcion == "5":
            ejemplo_estadisticas()
        elif opcion == "6":
            client_obj = APIClient()
            niveles = client_obj.obtener_niveles()
            print(f"\n📚 Niveles disponibles: {niveles.get('niveles', [])}")
        elif opcion == "7":
            ejemplo_health_check()
            ejemplo_prediccion_individual()
            ejemplo_prediccion_batch()
            ejemplo_clustering()
            ejemplo_estadisticas()
        elif opcion == "0":
            print("\n👋 ¡Hasta luego!")
            break
        else:
            print("\n❌ Opción inválida")
        
        input("\n⏸️ Presiona Enter para continuar...")


if __name__ == "__main__":
    print("\n🚀 Cliente de API - Análisis de Rendimiento Académico")
    print("="*70)
    print("\n⚠️ Asegúrate de que la API esté ejecutándose:")
    print("   python api.py")
    print("   o")
    print("   uvicorn api:app --reload")
    print("\n📍 URL de la API:", API_BASE_URL)
    print("="*70)
    
    # Verificar que la API esté disponible
    client = APIClient()
    health = client.health_check()
    
    if "error" in health:
        print("\n❌ La API no está disponible")
        print(f"   Error: {health['error']}")
        print("\n💡 Inicia la API primero con: python api.py")
    else:
        print("\n✅ API disponible y funcionando")
        
        # Ejecutar menú interactivo
        menu_interactivo()
