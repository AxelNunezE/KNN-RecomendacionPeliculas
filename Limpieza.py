import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SistemaRecomendacionPeliculas:
    def __init__(self, df_ratings, df_peliculas, df_usuarios=None):
        """
        Inicializar con datos ya limpios

        Args:
            df_ratings: DataFrame con columnas ['usuario_id', 'pelicula_id', 'rating']
            df_peliculas: DataFrame con información de películas
            df_usuarios: DataFrame con información de usuarios (opcional)
        """
        self.df_ratings = df_ratings
        self.df_peliculas = df_peliculas
        self.df_usuarios = df_usuarios
        self.modelo_knn = None
        self.matriz_ratings = None
        self.matriz_ratings_normalizada = None
        self.usuario_ids = None
        self.historial_metricas = []

    # SPRINT 2: DISEÑO Y ENTRENAMIENTO DEL MODELO
    def sprint_2_diseno_entrenamiento(self):
        """Sprint 2: Diseño y entrenamiento del modelo k-NN con datos ya limpios"""
        print("🎬 SPRINT 2: DISEÑO Y ENTRENAMIENTO DEL MODELO")
        print("="*50)

        # Preparar datos para el modelo
        self._preparar_datos_modelo()

        # Análisis de los datos preparados
        self._analizar_datos_preparados()

        # Diseñar y entrenar modelo
        self._entrenar_modelo_knn()

        # Evaluación inicial
        self._evaluar_modelo_inicial()

        print("✅ SPRINT 2 COMPLETADO: Modelo entrenado y evaluado")

    def _preparar_datos_modelo(self):
        """Preparar datos ya limpios para el modelo de recomendación"""
        print("🔧 PREPARANDO DATOS PARA EL MODELO...")

        # Crear matriz usuario-película
        self.matriz_ratings = self.df_ratings.pivot_table(
            index='usuario_id',
            columns='pelicula_id',
            values='rating',
            fill_value=0  # Llenar valores missing con 0
        )

        # Normalizar los ratings por usuario (centrar alrededor de su promedio)
        self.matriz_ratings_normalizada = self.matriz_ratings.apply(
            lambda x: x - x.mean() if x.mean() != 0 else x, axis=1
        )

        # Preparar datos para k-NN
        self.X = self.matriz_ratings_normalizada.values
        self.usuario_ids = self.matriz_ratings_normalizada.index.values

        print(f"✅ Datos preparados:")
        print(f"   - Matriz de ratings: {self.matriz_ratings.shape}")
        print(f"   - Usuarios: {len(self.usuario_ids)}")
        print(f"   - Películas: {len(self.matriz_ratings.columns)}")

    def _analizar_datos_preparados(self):
        """Análisis de los datos ya limpios"""
        print("\n📊 ANÁLISIS DE DATOS PREPARADOS:")

        # Estadísticas básicas
        ratings_no_cero = self.matriz_ratings.values[self.matriz_ratings.values != 0]
        if len(ratings_no_cero) > 0:
            print(f"   - Rating promedio (excluyendo ceros): {ratings_no_cero.mean():.2f}")
            print(f"   - Ratings no cero: {len(ratings_no_cero)}")

        densidad = (self.matriz_ratings != 0).sum().sum() / (self.matriz_ratings.shape[0] * self.matriz_ratings.shape[1])
        print(f"   - Densidad de la matriz: {densidad:.2%}")

        # Distribución de ratings por usuario
        ratings_por_usuario = (self.matriz_ratings != 0).sum(axis=1)
        print(f"   - Ratings promedio por usuario: {ratings_por_usuario.mean():.1f}")
        print(f"   - Usuario más activo: {ratings_por_usuario.max()} ratings")
        print(f"   - Usuario menos activo: {ratings_por_usuario.min()} ratings")

        # Visualización rápida
        if len(ratings_por_usuario) > 0:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            ratings_por_usuario.hist(bins=20, alpha=0.7)
            plt.title('Distribución de Ratings por Usuario')
            plt.xlabel('Número de Ratings')
            plt.ylabel('Frecuencia')

            plt.subplot(1, 2, 2)
            ratings_por_pelicula = (self.matriz_ratings != 0).sum(axis=0)
            ratings_por_pelicula.hist(bins=20, alpha=0.7)
            plt.title('Distribución de Ratings por Película')
            plt.xlabel('Número de Ratings')
            plt.ylabel('Frecuencia')

            plt.tight_layout()
            plt.show()

    def _entrenar_modelo_knn(self, n_vecinos=5):
        """Entrenar modelo k-NN para encontrar usuarios similares"""
        print(f"\n🤖 ENTRENANDO MODELO k-NN (k={n_vecinos})...")

        # Usar NearestNeighbors para encontrar usuarios similares
        self.modelo_knn = NearestNeighbors(
            n_neighbors=n_vecinos,
            metric='cosine',  # Similaridad coseno para datos sparse
            algorithm='brute'  # Funciona bien con datos de alta dimensionalidad
        )

        self.modelo_knn.fit(self.X)

        print("✅ Modelo k-NN entrenado exitosamente")
        print(f"   - Métrica: Cosine Similarity")
        print(f"   - Algoritmo: Brute Force")
        print(f"   - Vecinos: {n_vecinos}")
        print(f"   - Forma de datos: {self.X.shape}")

    def _evaluar_modelo_inicial(self):
        """Evaluación inicial del modelo"""
        print("\n📊 EVALUACIÓN INICIAL DEL MODELO:")

        # Probar con algunos usuarios de ejemplo
        if len(self.usuario_ids) > 0:
            usuario_ejemplo = self.usuario_ids[0]
            indice_usuario = np.where(self.usuario_ids == usuario_ejemplo)[0][0]

            # Encontrar vecinos más cercanos
            distancias, indices = self.modelo_knn.kneighbors([self.X[indice_usuario]])

            print(f"   - Usuario ejemplo: {usuario_ejemplo}")
            print(f"   - Vecinos más similares: {self.usuario_ids[indices[0]][1:]}")  # Excluir el mismo usuario
            print(f"   - Distancias: {distancias[0][1:].round(3)}")

        # Métrica de densidad de vecindarios
        self._calcular_metricas_vecindarios()

        # Guardar métricas iniciales
        self.historial_metricas.append({
            'sprint': 2,
            'k': 5,
            'similitud_promedio': self._calcular_similitud_promedio(),
            'usuarios_aislados': self._calcular_usuarios_aislados()
        })

    def _calcular_metricas_vecindarios(self):
        """Calcular métricas de calidad de los vecindarios"""
        if len(self.usuario_ids) > 100:  # Para datasets grandes, usar muestra
            indices_muestra = np.random.choice(len(self.usuario_ids), 100, replace=False)
            X_muestra = self.X[indices_muestra]
        else:
            X_muestra = self.X

        distancias, indices = self.modelo_knn.kneighbors(X_muestra)

        # Excluir el propio usuario (distancia 0)
        distancias_promedio = distancias[:, 1:].mean(axis=1)
        similitud_promedio = (1 - distancias[:, 1:]).mean()

        print(f"   - Distancia promedio entre vecinos: {distancias_promedio.mean():.3f}")
        print(f"   - Similitud promedio: {similitud_promedio:.3f}")
        print(f"   - Usuarios aislados (distancia > 0.8): {(distancias_promedio > 0.8).sum()}")

    def _calcular_similitud_promedio(self):
        """Calcular similitud promedio entre vecinos"""
        if len(self.usuario_ids) > 100:
            indices_muestra = np.random.choice(len(self.usuario_ids), 100, replace=False)
            X_muestra = self.X[indices_muestra]
        else:
            X_muestra = self.X

        distancias, _ = self.modelo_knn.kneighbors(X_muestra)
        return (1 - distancias[:, 1:]).mean()

    def _calcular_usuarios_aislados(self):
        """Calcular número de usuarios aislados"""
        if len(self.usuario_ids) > 100:
            indices_muestra = np.random.choice(len(self.usuario_ids), 100, replace=False)
            X_muestra = self.X[indices_muestra]
        else:
            X_muestra = self.X

        distancias, _ = self.modelo_knn.kneighbors(X_muestra)
        distancias_promedio = distancias[:, 1:].mean(axis=1)
        return (distancias_promedio > 0.8).sum()

    # SPRINT 3: OPTIMIZACIÓN Y AJUSTE DE HIPERPARÁMETROS
    def sprint_3_optimizacion(self):
        """Sprint 3: Optimización y ajuste de hiperparámetros"""
        print("\n🎬 SPRINT 3: OPTIMIZACIÓN Y AJUSTE")
        print("="*50)

        # Búsqueda de mejores hiperparámetros
        self._busqueda_hiperparametros()

        # Entrenar modelo optimizado
        self._entrenar_modelo_optimizado()

        # Validación del modelo optimizado
        self._validar_modelo_optimizado()

        print("✅ SPRINT 3 COMPLETADO: Modelo optimizado")

    def _busqueda_hiperparametros(self):
        """Búsqueda de mejores hiperparámetros para k-NN"""
        print("\n🔍 BÚSQUEDA DE MEJORES HIPERPARÁMETROS...")

        # Probar diferentes valores de k
        valores_k = [3, 5, 7, 10, 15, 20]
        metricas_k = []

        # Usar muestra para optimización si el dataset es grande
        if len(self.usuario_ids) > 200:
            indices_muestra = np.random.choice(len(self.usuario_ids), 200, replace=False)
            X_muestra = self.X[indices_muestra]
        else:
            X_muestra = self.X

        for k in valores_k:
            if k >= len(X_muestra):
                continue

            modelo_temp = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
            modelo_temp.fit(X_muestra)

            # Calcular métricas de calidad
            distancias, indices = modelo_temp.kneighbors(X_muestra)
            similitud_promedio = (1 - distancias[:, 1:]).mean()
            distancias_promedio = distancias[:, 1:].mean(axis=1)
            usuarios_aislados = (distancias_promedio > 0.8).sum()

            metricas_k.append({
                'k': k,
                'similitud_promedio': similitud_promedio,
                'usuarios_aislados': usuarios_aislados,
                'distancias_promedio': distancias_promedio.mean()
            })

        df_metricas = pd.DataFrame(metricas_k)

        # Seleccionar mejor k balanceando similitud y no aislamiento
        df_metricas['score'] = df_metricas['similitud_promedio'] * (1 - df_metricas['usuarios_aislados'] / len(X_muestra))
        self.mejor_k = df_metricas.loc[df_metricas['score'].idxmax()]['k']

        print("📊 Resultados de búsqueda de hiperparámetros:")
        print(df_metricas.round(4))
        print(f"🎯 Mejor k seleccionado: {int(self.mejor_k)}")

    def _entrenar_modelo_optimizado(self):
        """Entrenar modelo con los mejores hiperparámetros"""
        print(f"\n🔄 ENTRENANDO MODELO OPTIMIZADO (k={int(self.mejor_k)})...")

        self.modelo_knn_optimizado = NearestNeighbors(
            n_neighbors=int(self.mejor_k),
            metric='cosine',
            algorithm='brute'
        )

        self.modelo_knn_optimizado.fit(self.X)
        self.modelo_knn = self.modelo_knn_optimizado  # Usar el modelo optimizado

        print("✅ Modelo optimizado entrenado")

    def _validar_modelo_optimizado(self):
        """Validación del modelo optimizado"""
        print("\n📊 VALIDACIÓN DEL MODELO OPTIMIZADO:")

        # Calcular métricas con el nuevo modelo
        similitud_promedio = self._calcular_similitud_promedio()
        usuarios_aislados = self._calcular_usuarios_aislados()

        print(f"   - Similitud promedio: {similitud_promedio:.3f}")
        print(f"   - Usuarios aislados: {usuarios_aislados}")
        print(f"   - k óptimo: {int(self.mejor_k)}")

        # Guardar métricas
        self.historial_metricas.append({
            'sprint': 3,
            'k': self.mejor_k,
            'similitud_promedio': similitud_promedio,
            'usuarios_aislados': usuarios_aislados
        })

    # SPRINT 4: SISTEMA DE RECOMENDACIÓN COMPLETO
    def sprint_4_sistema_completo(self):
        """Sprint 4: Sistema de recomendación completo"""
        print("\n🎬 SPRINT 4: SISTEMA DE RECOMENDACIÓN COMPLETO")
        print("="*50)

        # Pruebas del sistema
        self._probar_sistema_recomendacion()

        # Métricas finales
        self._metricas_finales()

        print("✅ SPRINT 4 COMPLETADO: Sistema listo para producción")

    def _probar_sistema_recomendacion(self):
        """Probar el sistema de recomendación con usuarios de ejemplo"""
        print("\n🧪 PROBANDO SISTEMA DE RECOMENDACIÓN:")

        # Probar con varios usuarios de ejemplo
        usuarios_ejemplo = self.usuario_ids[:3] if len(self.usuario_ids) >= 3 else self.usuario_ids

        for usuario_id in usuarios_ejemplo:
            print(f"\n👤 RECOMENDACIONES PARA USUARIO {usuario_id}:")

            # Usuarios similares
            similares = self.encontrar_usuarios_similares(usuario_id, n_vecinos=3)
            print(f"   - Usuarios similares: {[sim[0] for sim in similares]}")

            # Recomendaciones
            recomendaciones = self.recomendar_para_usuario(usuario_id, n_recomendaciones=3)
            if recomendaciones:
                print(f"   - Películas recomendadas:")
                for pelicula_id, score in recomendaciones:
                    titulo = self._obtener_titulo_pelicula(pelicula_id)
                    print(f"     🎬 {titulo} (score: {score:.3f})")
            else:
                print(f"   - No hay suficientes datos para recomendaciones")

    def _obtener_titulo_pelicula(self, pelicula_id):
        """Obtener título de película por ID"""
        try:
            if hasattr(self.df_peliculas, 'columns'):
                if 'movie_title' in self.df_peliculas.columns:
                    return self.df_peliculas[self.df_peliculas['movie_id'] == pelicula_id]['movie_title'].values[0]
                elif 'title' in self.df_peliculas.columns:
                    return self.df_peliculas[self.df_peliculas['movie_id'] == pelicula_id]['title'].values[0]
            return f"Película {pelicula_id}"
        except:
            return f"Película {pelicula_id}"

    def _metricas_finales(self):
        """Mostrar métricas finales del proyecto"""
        print("\n📈 MÉTRICAS FINALES DEL PROYECTO:")

        if self.historial_metricas:
            df_metricas = pd.DataFrame(self.historial_metricas)
            print(df_metricas.round(4))

        # Métricas del sistema
        cobertura = self._calcular_cobertura()
        print(f"\n🎯 MÉTRICAS DEL SISTEMA:")
        print(f"   - Cobertura: {cobertura:.2%}")
        print(f"   - Usuarios atendidos: {len(self.usuario_ids)}")
        print(f"   - Películas en catálogo: {len(self.matriz_ratings.columns)}")
        print(f"   - k óptimo: {int(self.mejor_k)}")

    def _calcular_cobertura(self):
        """Calcular cobertura del sistema de recomendación"""
        if len(self.usuario_ids) > 50:
            usuarios_muestra = np.random.choice(self.usuario_ids, 50, replace=False)
        else:
            usuarios_muestra = self.usuario_ids

        peliculas_recomendables = set()

        for usuario_id in usuarios_muestra:
            recomendaciones = self.recomendar_para_usuario(usuario_id, 5)
            for pelicula_id, _ in recomendaciones:
                peliculas_recomendables.add(pelicula_id)

        return len(peliculas_recomendables) / len(self.matriz_ratings.columns)

    # FUNCIONES PRINCIPALES DEL SISTEMA
    def encontrar_usuarios_similares(self, usuario_id, n_vecinos=5):
        """Encontrar usuarios similares a un usuario dado"""
        if usuario_id not in self.usuario_ids:
            return []

        indice_usuario = np.where(self.usuario_ids == usuario_id)[0][0]
        distancias, indices = self.modelo_knn.kneighbors([self.X[indice_usuario]], n_neighbors=n_vecinos+1)

        # Excluir el propio usuario
        usuarios_similares = []
        for i in range(1, len(indices[0])):
            usuario_similar_id = self.usuario_ids[indices[0][i]]
            similitud = 1 - distancias[0][i]
            usuarios_similares.append((usuario_similar_id, similitud))

        return usuarios_similares

    def predecir_rating(self, usuario_id, pelicula_id):
        """Predecir rating que un usuario daría a una película"""
        if usuario_id not in self.usuario_ids or pelicula_id not in self.matriz_ratings.columns:
            return None

        indice_usuario = np.where(self.usuario_ids == usuario_id)[0][0]

        # Encontrar vecinos más cercanos
        distancias, indices = self.modelo_knn.kneighbors([self.X[indice_usuario]])

        # Calcular rating promedio ponderado por similitud
        rating_predicho = 0
        total_peso = 0

        for i in range(1, len(indices[0])):
            usuario_vecino_id = self.usuario_ids[indices[0][i]]
            similitud = 1 - distancias[0][i]

            # Rating del vecino para esta película
            rating_vecino = self.matriz_ratings.loc[usuario_vecino_id, pelicula_id]

            if rating_vecino > 0:
                rating_predicho += rating_vecino * similitud
                total_peso += similitud

        if total_peso > 0:
            return rating_predicho / total_peso
        else:
            # Si no hay vecinos que hayan calificado, usar promedio general de la película
            ratings_pelicula = self.df_ratings[self.df_ratings['pelicula_id'] == pelicula_id]['rating']
            return ratings_pelicula.mean() if len(ratings_pelicula) > 0 else 3.0

    def recomendar_para_usuario(self, usuario_id, n_recomendaciones=5):
        """Generar recomendaciones para un usuario"""
        if usuario_id not in self.usuario_ids:
            return []

        # Películas que el usuario ya ha visto
        indice_usuario = np.where(self.usuario_ids == usuario_id)[0][0]
        peliculas_vistas = self.matriz_ratings.columns[
            self.matriz_ratings.iloc[indice_usuario] > 0
        ].tolist()

        # Predecir ratings para películas no vistas
        predicciones = []
        for pelicula_id in self.matriz_ratings.columns:
            if pelicula_id not in peliculas_vistas:
                rating_predicho = self.predecir_rating(usuario_id, pelicula_id)
                if rating_predicho is not None and rating_predicho > 0:
                    predicciones.append((pelicula_id, rating_predicho))

        # Ordenar por rating predicho y devolver top N
        predicciones.sort(key=lambda x: x[1], reverse=True)
        return predicciones[:n_recomendaciones]

    def recomendar_por_genero(self, generos_preferidos, n_recomendaciones=10):
        """
        Recomienda películas basadas en una lista de géneros preferidos.

        Args:
            generos_preferidos: Lista de strings con los nombres de los géneros.
            n_recomendaciones: Número de películas a recomendar.

        Returns:
            Lista de tuplas (pelicula_id, titulo) de las películas recomendadas.
        """
        print(f"\n🔍 Buscando películas en los géneros: {', '.join(generos_preferidos)}")

        # Filtrar películas que contienen al menos uno de los géneros
        peliculas_filtradas = self.df_peliculas[
            self.df_peliculas['movie_genre'].str.contains('|'.join(generos_preferidos), case=False, na=False)
        ]

        if peliculas_filtradas.empty:
            print("❌ No se encontraron películas para los géneros especificados.")
            return []

        # Ordenar por popularidad y seleccionar las top N
        recomendaciones = peliculas_filtradas.sort_values(by='movie_popularity', ascending=False).head(n_recomendaciones)

        # Formatear salida
        lista_recomendaciones = []
        for index, row in recomendaciones.iterrows():
            lista_recomendaciones.append((row['movie_id'], row['movie_title']))

        return lista_recomendaciones


# FUNCIÓN PARA CARGAR Y PREPARAR DATOS DEL CSV ESPECÍFICO
def cargar_y_preparar_datos_csv(archivo_csv):
    """
    Cargar y preparar datos específicos del archivo CSV proporcionado

    Args:
        archivo_csv: Ruta al archivo CSV con datos de películas
    """
    print(f"📂 CARGANDO DATOS DESDE: {archivo_csv}")

    # Cargar el CSV
    df_peliculas = pd.read_csv(archivo_csv)

    print(f"✅ CSV cargado: {len(df_peliculas)} películas encontradas")
    print(f"📊 Columnas disponibles: {list(df_peliculas.columns)}")

    # Mostrar información básica del dataset
    print("\n🎬 INFORMACIÓN DEL DATASET:")
    print("-" * 40)
    print(f"Películas únicas: {df_peliculas['movie_id'].nunique()}")
    print(f"Géneros: {df_peliculas['movie_genre'].nunique()}")
    print(f"Idiomas: {df_peliculas['movie_language'].nunique()}")

    # Como el CSV no tiene ratings, necesitamos simular datos de ratings
    # para poder ejecutar el sistema de recomendación
    df_ratings = simular_datos_ratings(df_peliculas)

    return df_ratings, df_peliculas

def simular_datos_ratings(df_peliculas, n_usuarios=100, max_ratings_por_usuario=20):
    """
    Simular datos de ratings ya que el CSV original no los tiene

    Args:
        df_peliculas: DataFrame con información de películas
        n_usuarios: Número de usuarios a simular
        max_ratings_por_usuario: Máximo de ratings por usuario
    """
    print(f"\n🎲 SIMULANDO DATOS DE RATINGS...")

    ratings_data = []
    pelicula_ids = df_peliculas['movie_id'].unique()

    for usuario_id in range(1, n_usuarios + 1):
        # Cada usuario califica un número aleatorio de películas
        n_ratings = np.random.randint(5, max_ratings_por_usuario + 1)
        peliculas_calificadas = np.random.choice(pelicula_ids, n_ratings, replace=False)

        for pelicula_id in peliculas_calificadas:
            # Generar rating basado en popularidad y algo de aleatoriedad
            pelicula_info = df_peliculas[df_peliculas['movie_id'] == pelicula_id].iloc[0]

            # Base del rating basada en popularidad y votos
            base_rating = 5.0  # Rating base

            # Ajustar por popularidad si está disponible
            if 'movie_popularity' in df_peliculas.columns and not pd.isna(pelicula_info['movie_popularity']):
                base_rating += min(pelicula_info['movie_popularity'] / 50, 3.0)

            if 'movie_vote' in df_peliculas.columns and not pd.isna(pelicula_info['movie_vote']):
                base_rating += (pelicula_info['movie_vote'] - 5.0) / 2

            # Añadir variabilidad personal
            rating = max(1.0, min(5.0, base_rating + np.random.normal(0, 1)))
            rating = round(rating * 2) / 2  # Redondear a 0.5

            ratings_data.append({
                'usuario_id': usuario_id,
                'pelicula_id': pelicula_id,
                'rating': rating
            })

    df_ratings = pd.DataFrame(ratings_data)

    print(f"✅ Ratings simulados:")
    print(f"   - Usuarios: {n_usuarios}")
    print(f"   - Ratings totales: {len(df_ratings)}")
    print(f"   - Rating promedio: {df_ratings['rating'].mean():.2f}")

    return df_ratings

# FUNCIÓN PARA EJECUTAR EL PROYECTO COMPLETO CON EL CSV ESPECÍFICO
def ejecutar_proyecto_con_csv(archivo_csv):
    """
    Ejecutar todos los sprints del proyecto con el CSV específico

    Args:
        archivo_csv: Ruta al archivo CSV con datos de películas
    """
    print("🚀 INICIANDO SISTEMA DE RECOMENDACIÓN CON CSV ESPECÍFICO")
    print("="*60)

    # Cargar y preparar datos del CSV
    df_ratings, df_peliculas = cargar_y_preparar_datos_csv(archivo_csv)

    print("\n📊 DATOS PREPARADOS:")
    print(f"   - Ratings: {len(df_ratings)} registros")
    print(f"   - Películas: {len(df_peliculas)} en catálogo")
    print(f"   - Usuarios únicos: {df_ratings['usuario_id'].nunique()}")
    print("="*60)

    # Crear instancia del sistema con datos preparados
    sistema = SistemaRecomendacionPeliculas(df_ratings, df_peliculas)

    # Ejecutar sprints
    sistema.sprint_2_diseno_entrenamiento()
    sistema.sprint_3_optimizacion()
    sistema.sprint_4_sistema_completo()

    print("\n🎉 ¡SISTEMA DE RECOMENDACIÓN COMPLETADO!")
    print("="*60)

    return sistema

# EJECUTAR CON EL CSV ESPECÍFICO
if __name__ == "__main__":
    # Reemplaza con la ruta correcta a tu archivo CSV
    archivo_csv = "Movies Recommendation_limpiado_ordenado.csv"

    try:
        # Ejecutar el sistema con el CSV específico
        sistema = ejecutar_proyecto_con_csv(archivo_csv)

        # Ejemplo adicional: probar recomendaciones para un usuario específico
        print("\n🔍 PRUEBA ADICIONAL: Recomendaciones detalladas")
        print("-" * 50)

        if hasattr(sistema, 'usuario_ids') and len(sistema.usuario_ids) > 0:
            usuario_prueba = sistema.usuario_ids[0]
            print(f"\n🎯 RECOMENDACIONES DETALLADAS PARA USUARIO {usuario_prueba}:")

            recomendaciones = sistema.recomendar_para_usuario(usuario_prueba, n_recomendaciones=5)
            for i, (pelicula_id, score) in enumerate(recomendaciones, 1):
                titulo = sistema._obtener_titulo_pelicula(pelicula_id)
                print(f"   {i}. {titulo} - Score: {score:.3f}")

    except FileNotFoundError:
        print(f"❌ Error: No se pudo encontrar el archivo '{archivo_csv}'")
        print("💡 Asegúrate de que el archivo esté en el mismo directorio o proporciona la ruta completa")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
