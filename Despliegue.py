import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SistemaRecomendacionInteractivo:
    def __init__(self, df_peliculas):
        """
        Sistema de recomendación interactivo con entrada de usuario real

        Args:
            df_peliculas: DataFrame con información de películas del CSV
        """
        self.df_peliculas = df_peliculas
        self.df_ratings = None
        self.modelo_knn = None
        self.matriz_ratings = None
        self.usuario_ids = None
        self.usuario_actual_id = None
        self.generos_disponibles = self._obtener_generos_disponibles()

    def _obtener_generos_disponibles(self):
        """Obtener lista de géneros únicos disponibles en el dataset"""
        todos_generos = set()
        for genero_str in self.df_peliculas['movie_genre'].dropna():
            # Los géneros pueden venir como "Action,Adventure,Sci-Fi" o individuales
            generos = [g.strip() for g in str(genero_str).split(',')]
            todos_generos.update(generos)
        return sorted(list(todos_generos))

    def preparar_datos_base(self):
        """Preparar datos base con ratings simulados de usuarios existentes"""
        print("🔄 PREPARANDO DATOS BASE...")

        # Simular datos de ratings de usuarios base
        self.df_ratings = self._simular_ratings_usuarios_base()

        # Crear matriz usuario-película
        self.matriz_ratings = self.df_ratings.pivot_table(
            index='usuario_id',
            columns='pelicula_id',
            values='rating',
            fill_value=0
        )

        # Normalizar
        self.matriz_ratings_normalizada = self.matriz_ratings.apply(
            lambda x: x - x.mean() if x.mean() != 0 else x, axis=1
        )

        self.X = self.matriz_ratings_normalizada.values
        self.usuario_ids = self.matriz_ratings_normalizada.index.values

        print(f"✅ Datos base preparados: {len(self.usuario_ids)} usuarios, {len(self.matriz_ratings.columns)} películas")

    def _simular_ratings_usuarios_base(self, n_usuarios_base=100):
        """Simular ratings para usuarios base del sistema con gustos realistas"""
        ratings_data = []
        pelicula_ids = self.df_peliculas['movie_id'].unique()

        # Crear perfiles de usuarios base con gustos definidos
        generos_peliculas = self._mapear_generos_peliculas()

        for usuario_id in range(1, n_usuarios_base + 1):
            # Asignar preferencias de género aleatorias pero realistas
            generos_favoritos = np.random.choice(
                self.generos_disponibles,
                size=np.random.randint(2, 5),
                replace=False
            )

            # Calificar películas según preferencias
            n_ratings = np.random.randint(20, 50)
            peliculas_calificadas = []

            # Priorizar películas de géneros favoritos (60% de sus ratings)
            for genero in generos_favoritos:
                if genero in generos_peliculas:
                    peliculas_genero = generos_peliculas[genero]
                    n_peliculas_genero = min(len(peliculas_genero), int(n_ratings * 0.6 / len(generos_favoritos)))
                    if n_peliculas_genero > 0:
                        seleccionadas = np.random.choice(peliculas_genero, n_peliculas_genero, replace=False)
                        peliculas_calificadas.extend(seleccionadas)

            # Completar con películas aleatorias (40% restante)
            peliculas_restantes = [p for p in pelicula_ids if p not in peliculas_calificadas]
            n_restantes = n_ratings - len(peliculas_calificadas)
            if n_restantes > 0 and len(peliculas_restantes) > 0:
                peliculas_calificadas.extend(
                    np.random.choice(peliculas_restantes, n_restantes, replace=False)
                )

            # Asignar ratings realistas basados en preferencias
            for pelicula_id in peliculas_calificadas:
                pelicula_info = self.df_peliculas[self.df_peliculas['movie_id'] == pelicula_id].iloc[0]

                # Rating base basado en género favorito
                generos_pelicula = self._obtener_generos_pelicula(pelicula_info['movie_genre'])
                generos_comunes = set(generos_pelicula) & set(generos_favoritos)

                if generos_comunes:
                    # Alta probabilidad de rating alto si coincide con género favorito
                    base_rating = 4.0 + np.random.random() * 1.0  # 4.0-5.0
                else:
                    # Rating más neutral para otros géneros
                    base_rating = 3.0 + np.random.random() * 1.5  # 3.0-4.5

                # Ajustar por popularidad de la película
                if 'movie_vote' in pelicula_info and not pd.isna(pelicula_info['movie_vote']):
                    ajuste_popularidad = (pelicula_info['movie_vote'] - 6.0) / 4.0
                    base_rating += ajuste_popularidad

                rating = max(1.0, min(5.0, base_rating))
                rating = round(rating * 2) / 2  # Redondear a 0.5

                ratings_data.append({
                    'usuario_id': usuario_id,
                    'pelicula_id': pelicula_id,
                    'rating': rating
                })

        return pd.DataFrame(ratings_data)

    def _mapear_generos_peliculas(self):
        """Crear mapeo de género a lista de películas"""
        generos_peliculas = {}
        for _, pelicula in self.df_peliculas.iterrows():
            generos = self._obtener_generos_pelicula(pelicula['movie_genre'])
            for genero in generos:
                if genero not in generos_peliculas:
                    generos_peliculas[genero] = []
                generos_peliculas[genero].append(pelicula['movie_id'])
        return generos_peliculas

    def _obtener_generos_pelicula(self, genero_str):
        """Extraer lista de géneros de una película"""
        if pd.isna(genero_str):
            return ['Unknown']
        return [g.strip() for g in str(genero_str).split(',')]

    def _obtener_genero_principal(self, genero_str):
        """Obtener el género principal de una película"""
        generos = self._obtener_generos_pelicula(genero_str)
        return generos[0] if generos else 'Unknown'

    def entrenar_modelo(self, k_vecinos=10):
        """Entrenar el modelo KNN"""
        print(f"🤖 ENTRENANDO MODELO CON k={k_vecinos}...")

        self.modelo_knn = NearestNeighbors(
            n_neighbors=k_vecinos,
            metric='cosine',
            algorithm='brute'
        )
        self.modelo_knn.fit(self.X)
        print("✅ Modelo entrenado exitosamente")

    # 🎯 NUEVA SECCIÓN INTERACTIVA - ENTRADA DE USUARIO REAL
    def preguntar_preferencias_usuario(self):
        """Sistema interactivo para preguntar preferencias al usuario"""
        print("\n" + "="*60)
        print("🎬 SISTEMA DE RECOMENDACIÓN PERSONALIZADO")
        print("="*60)
        print("¡Hola! Vamos a encontrar películas perfectas para ti.")
        print("\nPrimero, cuéntame sobre tus gustos cinematográficos...")

        # Mostrar géneros disponibles
        print(f"\n📚 GÉNEROS DISPONIBLES ({len(self.generos_disponibles)}):")
        for i, genero in enumerate(self.generos_disponibles, 1):
            print(f"   {i:2d}. {genero}")

        # Preguntar por géneros favoritos
        print("\n🔍 SELECCIÓN DE GÉNEROS FAVORITOS")
        print("¿Cuáles son tus géneros de película favoritos?")
        print("(Ingresa los números separados por comas, ej: 1,3,5)")

        generos_seleccionados = []
        while not generos_seleccionados:
            try:
                entrada = input("👉 Tus géneros favoritos: ").strip()
                if entrada:
                    numeros = [int(x.strip()) for x in entrada.split(',')]
                    generos_seleccionados = [self.generos_disponibles[n-1] for n in numeros
                                           if 1 <= n <= len(self.generos_disponibles)]

                if not generos_seleccionados:
                    print("❌ Por favor ingresa al menos un género válido")
            except (ValueError, IndexError):
                print("❌ Formato incorrecto. Usa números separados por comas.")

        print(f"\n✅ Tus géneros favoritos: {', '.join(generos_seleccionados)}")

        # Preguntar por nivel de rating mínimo
        print("\n⭐ PREFERENCIA DE CALIDAD")
        print("¿Qué rating mínimo prefieres en las películas?")
        print("1. ⭐⭐⭐⭐⭐ Excelente (8.0+)")
        print("2. ⭐⭐⭐⭐ Muy bueno (7.0+)")
        print("3. ⭐⭐⭐ Bueno (6.0+)")
        print("4. ⭐⭐ Regular (5.0+)")
        print("5. Cualquier calidad")

        opcion_rating = input("👉 Tu elección (1-5): ").strip()
        umbrales = {'1': 8.0, '2': 7.0, '3': 6.0, '4': 5.0, '5': 0.0}
        rating_minimo = umbrales.get(opcion_rating, 0.0)

        # Preguntar por década preferida (opcional)
        print("\n📅 PREFERENCIA TEMPORAL (opcional)")
        print("¿Tienes preferencia por alguna década?")
        print("1. 2000s - Actualidad")
        print("2. 1990s")
        print("3. 1980s")
        print("4. 1970s o anterior")
        print("5. Cualquier época")

        opcion_decada = input("👉 Tu elección (1-5): ").strip()

        # Crear perfil de usuario
        perfil_usuario = {
            'generos_favoritos': generos_seleccionados,
            'rating_minimo': rating_minimo,
            'decada_preferida': opcion_decada,
            'timestamp': pd.Timestamp.now()
        }

        return perfil_usuario

    def crear_usuario_y_ratings(self, perfil_usuario):
        """Crear un nuevo usuario en el sistema basado en sus preferencias"""
        # El nuevo usuario será el siguiente ID disponible
        nuevo_usuario_id = max(self.usuario_ids) + 1 if len(self.usuario_ids) > 0 else 1
        self.usuario_actual_id = nuevo_usuario_id

        print(f"\n👤 CREANDO TU PERFIL (Usuario #{nuevo_usuario_id})...")

        # Generar ratings basados en preferencias
        nuevos_ratings = []
        generos_peliculas = self._mapear_generos_peliculas()

        # Para cada película, predecir rating basado en preferencias
        for pelicula_id in self.matriz_ratings.columns:
            pelicula_info = self.df_peliculas[self.df_peliculas['movie_id'] == pelicula_id].iloc[0]

            # Calcular rating basado en coincidencia de géneros
            generos_pelicula = self._obtener_generos_pelicula(pelicula_info['movie_genre'])
            generos_comunes = set(generos_pelicula) & set(perfil_usuario['generos_favoritos'])

            if generos_comunes:
                # Alta probabilidad de rating alto si coincide con género favorito
                base_rating = 4.0 + (len(generos_comunes) * 0.3)  # Más géneros en común = mejor rating
            else:
                # Rating bajo para géneros no favoritos
                base_rating = 2.0 + np.random.random() * 1.0

            # Ajustar por calidad de la película
            if 'movie_vote' in pelicula_info and not pd.isna(pelicula_info['movie_vote']):
                ajuste_calidad = (pelicula_info['movie_vote'] - 6.0) / 3.0
                base_rating += ajuste_calidad

            # Ajustar por popularidad
            if 'movie_popularity' in pelicula_info and not pd.isna(pelicula_info['movie_popularity']):
                ajuste_popularidad = min(pelicula_info['movie_popularity'] / 50, 1.0)
                base_rating += ajuste_popularidad

            rating = max(1.0, min(5.0, base_rating))
            rating = round(rating * 2) / 2  # Redondear a 0.5

            # Solo incluir ratings significativos (>2.5)
            if rating > 2.5:
                nuevos_ratings.append({
                    'usuario_id': nuevo_usuario_id,
                    'pelicula_id': pelicula_id,
                    'rating': rating
                })

        # Agregar nuevos ratings al dataset
        df_nuevos_ratings = pd.DataFrame(nuevos_ratings)
        self.df_ratings = pd.concat([self.df_ratings, df_nuevos_ratings], ignore_index=True)

        print(f"✅ Perfil creado con {len(nuevos_ratings)} preferencias automáticas")
        return nuevo_usuario_id

    def generar_recomendaciones_personalizadas(self, usuario_id, n_recomendaciones=10):
        """Generar recomendaciones personalizadas para el usuario"""
        print(f"\n🎯 GENERANDO TUS {n_recomendaciones} RECOMENDACIONES PERSONALIZADAS...")

        # Actualizar matriz con el nuevo usuario
        self._actualizar_matriz_con_nuevo_usuario()

        # Encontrar usuarios similares
        usuarios_similares = self._encontrar_usuarios_similares(usuario_id)

        # Generar recomendaciones
        recomendaciones = self._recomendar_para_usuario(usuario_id, n_recomendaciones)

        return recomendaciones, usuarios_similares

    def _actualizar_matriz_con_nuevo_usuario(self):
        """Actualizar la matriz de ratings con el nuevo usuario"""
        self.matriz_ratings = self.df_ratings.pivot_table(
            index='usuario_id',
            columns='pelicula_id',
            values='rating',
            fill_value=0
        )

        self.matriz_ratings_normalizada = self.matriz_ratings.apply(
            lambda x: x - x.mean() if x.mean() != 0 else x, axis=1
        )

        self.X = self.matriz_ratings_normalizada.values
        self.usuario_ids = self.matriz_ratings_normalizada.index.values

        # Re-entrenar modelo con el nuevo usuario
        self.entrenar_modelo(k_vecinos=min(10, len(self.usuario_ids)-1))

    def _encontrar_usuarios_similares(self, usuario_id, n_vecinos=5):
        """Encontrar usuarios similares al usuario actual"""
        if usuario_id not in self.usuario_ids:
            return []

        indice_usuario = np.where(self.usuario_ids == usuario_id)[0][0]
        distancias, indices = self.modelo_knn.kneighbors([self.X[indice_usuario]], n_neighbors=n_vecinos+1)

        usuarios_similares = []
        for i in range(1, len(indices[0])):
            usuario_similar_id = self.usuario_ids[indices[0][i]]
            similitud = 1 - distancias[0][i]
            usuarios_similares.append((usuario_similar_id, similitud))

        return usuarios_similares

    def _recomendar_para_usuario(self, usuario_id, n_recomendaciones=10):
        """Generar recomendaciones para el usuario"""
        if usuario_id not in self.usuario_ids:
            return []

        # Películas que el usuario ya ha "visto" (con rating alto)
        indice_usuario = np.where(self.usuario_ids == usuario_id)[0][0]
        peliculas_vistas = self.matriz_ratings.columns[
            self.matriz_ratings.iloc[indice_usuario] > 3.0  # Considerar como vistas las >3.0
        ].tolist()

        # Predecir ratings para películas no vistas
        predicciones = []
        for pelicula_id in self.matriz_ratings.columns:
            if pelicula_id not in peliculas_vistas:
                rating_predicho = self._predecir_rating(usuario_id, pelicula_id)
                if rating_predicho is not None and rating_predicho > 3.5:  # Solo recomendables
                    predicciones.append((pelicula_id, rating_predicho))

        # Ordenar y devolver top N
        predicciones.sort(key=lambda x: x[1], reverse=True)
        return predicciones[:n_recomendaciones]

    def _predecir_rating(self, usuario_id, pelicula_id):
        """Predecir rating usando KNN"""
        if usuario_id not in self.usuario_ids or pelicula_id not in self.matriz_ratings.columns:
            return None

        indice_usuario = np.where(self.usuario_ids == usuario_id)[0][0]
        distancias, indices = self.modelo_knn.kneighbors([self.X[indice_usuario]])

        rating_predicho = 0
        total_peso = 0

        for i in range(1, len(indices[0])):
            usuario_vecino_id = self.usuario_ids[indices[0][i]]
            similitud = 1 - distancias[0][i]
            rating_vecino = self.matriz_ratings.loc[usuario_vecino_id, pelicula_id]

            if rating_vecino > 0:
                rating_predicho += rating_vecino * similitud
                total_peso += similitud

        return rating_predicho / total_peso if total_peso > 0 else None

    def mostrar_recomendaciones(self, recomendaciones, usuarios_similares, perfil_usuario):
        """Mostrar las recomendaciones de forma atractiva"""
        print("\n" + "="*70)
        print("🎉 ¡TUS 10 RECOMENDACIONES PERSONALIZADAS!")
        print("="*70)

        # Mostrar resumen del perfil
        print(f"\n📊 TU PERFIL:")
        print(f"   • Géneros favoritos: {', '.join(perfil_usuario['generos_favoritos'])}")
        print(f"   • Usuarios similares encontrados: {len(usuarios_similares)}")
        if usuarios_similares:
            similitud_promedio = np.mean([sim for _, sim in usuarios_similares])
            print(f"   • Similitud promedio: {similitud_promedio:.1%}")

        print(f"\n🎬 RECOMENDACIONES:")

        for i, (pelicula_id, score) in enumerate(recomendaciones, 1):
            pelicula_info = self.df_peliculas[self.df_peliculas['movie_id'] == pelicula_id].iloc[0]

            # Obtener información de la película
            titulo = pelicula_info.get('movie_title', f'Película {pelicula_id}')
            genero = pelicula_info.get('movie_genre', 'Desconocido')
            rating = pelicula_info.get('movie_vote', 'N/A')
            año = str(pelicula_info.get('movie_release_date', 'N/A'))[:4] if pd.notna(pelicula_info.get('movie_release_date')) else 'N/A'

            print(f"\n{i:2d}. 🎭 {titulo}")
            print(f"    📅 Año: {año} | ⭐ Rating: {rating}/10 | 🎞️ Género: {genero}")
            print(f"    🔥 Score de recomendación: {score:.3f}")

            # Mostrar sinopsis si está disponible
            if 'movie_overview' in pelicula_info and pd.notna(pelicula_info['movie_overview']):
                sinopsis = pelicula_info['movie_overview']
                if len(sinopsis) > 120:
                    sinopsis = sinopsis[:117] + "..."
                print(f"    📖 {sinopsis}")

        print("\n" + "="*70)
        print("💡 Tip: Basado en usuarios con gustos similares a los tuyos")
        print("="*70)

# FUNCIÓN PRINCIPAL DE DESPLIEGUE
def ejecutar_sistema_recomendacion_interactivo(archivo_csv):
    """
    Ejecutar el sistema completo de recomendación interactivo
    """
    print("🚀 INICIANDO SISTEMA DE RECOMENDACIÓN INTERACTIVO")
    print("="*60)

    try:
        # Cargar datos
        df_peliculas = pd.read_csv(archivo_csv)
        print(f"✅ CSV cargado: {len(df_peliculas)} películas")

        # Crear sistema
        sistema = SistemaRecomendacionInteractivo(df_peliculas)

        # Preparar datos base
        sistema.preparar_datos_base()

        # Entrenar modelo inicial
        sistema.entrenar_modelo()

        # 🎯 INTERACCIÓN CON EL USUARIO REAL
        perfil_usuario = sistema.preguntar_preferencias_usuario()

        # Crear usuario en el sistema
        usuario_id = sistema.crear_usuario_y_ratings(perfil_usuario)

        # Generar recomendaciones
        recomendaciones, usuarios_similares = sistema.generar_recomendaciones_personalizadas(
            usuario_id, n_recomendaciones=10
        )

        # Mostrar resultados
        sistema.mostrar_recomendaciones(recomendaciones, usuarios_similares, perfil_usuario)

        return sistema, recomendaciones

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo '{archivo_csv}'")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

# EJECUTAR EL SISTEMA
if __name__ == "__main__":
    archivo_csv = "Movies Recommendation_limpiado_ordenado.csv"

    print("🎬 BIENVENIDO AL SISTEMA DE RECOMENDACIÓN DE PELÍCULAS")
    print("="*60)

    sistema, recomendaciones = ejecutar_sistema_recomendacion_interactivo(archivo_csv)

    # Opción para nuevas recomendaciones
    if sistema and recomendaciones:
        print("\n🔄 ¿Quieres probar con otro perfil? (s/n): ")
        respuesta = input().strip().lower()
        if respuesta == 's':
            ejecutar_sistema_recomendacion_interactivo(archivo_csv)
