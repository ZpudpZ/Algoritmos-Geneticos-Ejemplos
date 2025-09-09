#Ejemplo para Optimización de Hiperparámetros (Hyperparameter Optimization)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Preparación de los datos
iris = load_iris()
X = iris.data
y = iris.target
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Parámetros del algoritmo genético
TAMANO_POBLACION = 50
GENERACIONES = 100
TASA_MUTACION = 0.05


def crear_individuo():
    n_neighbors = np.random.randint(1, 21)
    p = np.random.randint(1, 3)
    return [n_neighbors, p]


def calcular_aptitud(individuo):
    n_neighbors, p = individuo

    if n_neighbors < 1 or p not in [1, 2]:
        return 0

    modelo = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)
    modelo.fit(X_entrenamiento, y_entrenamiento)
    predicciones = modelo.predict(X_prueba)

    return accuracy_score(y_prueba, predicciones)


def seleccionar_padres(poblacion, puntajes_aptitud):
    total_aptitud = sum(puntajes_aptitud)
    if total_aptitud == 0:
        total_aptitud = 1

    probabilidades = [score / total_aptitud for score in puntajes_aptitud]
    parent_indices = np.random.choice(range(TAMANO_POBLACION), size=2, p=probabilidades, replace=False)

    return poblacion[parent_indices[0]], poblacion[parent_indices[1]]


def cruzar(padre1, padre2):
    crossover_point = np.random.randint(1, len(padre1))
    hijo1 = padre1[:crossover_point] + padre2[crossover_point:]
    hijo2 = padre2[:crossover_point] + padre1[crossover_point:]
    return hijo1, hijo2


def mutar(individuo):
    if np.random.rand() < TASA_MUTACION:
        individuo[0] = np.random.randint(1, 21)
    if np.random.rand() < TASA_MUTACION:
        individuo[1] = np.random.randint(1, 3)
    return individuo


# 3. Bucle principal del algoritmo genético
poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]
mejor_aptitud_por_generacion = []
aptitud_promedio_por_generacion = []
mejor_solucion_por_generacion = []

for generacion in range(GENERACIONES):
    puntajes_aptitud = [calcular_aptitud(individuo) for individuo in poblacion]

    indice_mejor_individuo = np.argmax(puntajes_aptitud)
    mejor_individuo_actual = poblacion[indice_mejor_individuo]
    mejor_aptitud = puntajes_aptitud[indice_mejor_individuo]
    aptitud_promedio = np.mean(puntajes_aptitud)

    mejor_aptitud_por_generacion.append(mejor_aptitud)
    aptitud_promedio_por_generacion.append(aptitud_promedio)
    mejor_solucion_por_generacion.append(mejor_individuo_actual)

    print(
        f"Generación {generacion + 1}/{GENERACIONES}: Mejor Precisión = {mejor_aptitud:.4f}, Mejor Hiperparámetros = {mejor_individuo_actual}, Precisión Promedio = {aptitud_promedio:.4f}")

    nueva_poblacion = []

    nueva_poblacion.append(poblacion[indice_mejor_individuo])

    while len(nueva_poblacion) < TAMANO_POBLACION:
        padre1, padre2 = seleccionar_padres(poblacion, puntajes_aptitud)
        hijo1, hijo2 = cruzar(padre1, padre2)

        hijo_mutado1 = mutar(hijo1)
        hijo_mutado2 = mutar(hijo2)

        nueva_poblacion.append(hijo_mutado1)
        if len(nueva_poblacion) < TAMANO_POBLACION:
            nueva_poblacion.append(hijo_mutado2)

    poblacion = nueva_poblacion

# 4. Resultados y visualización final
mejor_solucion_final = mejor_solucion_por_generacion[-1]
mejor_precision_final = mejor_aptitud_por_generacion[-1]

print("\n--- Resultados Finales ---")
print(f"Mejores hiperparámetros encontrados: {mejor_solucion_final}")
print(f"Precisión del modelo con estos hiperparámetros: {mejor_precision_final:.4f}")

# Gráfico de la evolución de la precisión
plt.figure(figsize=(10, 6))
plt.plot(range(GENERACIONES), mejor_aptitud_por_generacion, label='Mejor precisión por generación', color='blue',
         marker='o', markersize=4)
plt.plot(range(GENERACIONES), aptitud_promedio_por_generacion, label='Precisión promedio por generación',
         color='orange',
         linestyle='--')
plt.title('Evolución de la precisión del modelo a lo largo de las generaciones')
plt.xlabel('Generación')
plt.ylabel('Precisión (Aptitud)')
plt.grid(True)
plt.legend()
plt.show()