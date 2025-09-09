#Ejemplo de Algoritmo Genético para Feature Selection
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Preparación de los datos
iris = load_iris()
X = iris.data
y = iris.target
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Parámetros del algoritmo genético
TAMANO_POBLACION = 50
LONGITUD_CROMOSOMA = X.shape[1]
GENERACIONES = 100
TASA_MUTACION = 0.05


def crear_individuo():
    return np.random.randint(2, size=LONGITUD_CROMOSOMA)


def calcular_aptitud(individuo):
    caracteristicas_seleccionadas = np.where(individuo == 1)[0]

    if len(caracteristicas_seleccionadas) == 0:
        return 0

    X_entrenamiento_subconjunto = X_entrenamiento[:, caracteristicas_seleccionadas]
    X_prueba_subconjunto = X_prueba[:, caracteristicas_seleccionadas]

    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_entrenamiento_subconjunto, y_entrenamiento)

    predicciones = modelo.predict(X_prueba_subconjunto)

    return accuracy_score(y_prueba, predicciones)


def seleccionar_padres(poblacion, puntajes_aptitud):
    total_aptitud = sum(puntajes_aptitud)

    if total_aptitud == 0:
        total_aptitud = 1

    probabilidades = [puntaje / total_aptitud for puntaje in puntajes_aptitud]

    indices_padres = np.random.choice(range(TAMANO_POBLACION), size=2, p=probabilidades, replace=False)

    return poblacion[indices_padres[0]], poblacion[indices_padres[1]]


def cruzar(padre1, padre2):
    punto_cruce = np.random.randint(1, LONGITUD_CROMOSOMA - 1)
    hijo1 = np.concatenate((padre1[:punto_cruce], padre2[punto_cruce:]))
    hijo2 = np.concatenate((padre2[:punto_cruce], padre1[punto_cruce:]))
    return hijo1, hijo2


def mutar(individuo):
    for i in range(LONGITUD_CROMOSOMA):
        if np.random.rand() < TASA_MUTACION:
            individuo[i] = 1 - individuo[i]
    return individuo

# 3. Bucle principal del algoritmo genético
poblacion = [crear_individuo() for _ in range(TAMANO_POBLACION)]
mejor_aptitud_por_generacion = []
aptitud_promedio_por_generacion = []

for generacion in range(GENERACIONES):
    puntajes_aptitud = [calcular_aptitud(individuo) for individuo in poblacion]

    indice_mejor_individuo = np.argmax(puntajes_aptitud)
    mejor_individuo_actual = poblacion[indice_mejor_individuo]
    mejor_aptitud = puntajes_aptitud[indice_mejor_individuo]
    aptitud_promedio = np.mean(puntajes_aptitud)

    mejor_aptitud_por_generacion.append(mejor_aptitud)
    aptitud_promedio_por_generacion.append(aptitud_promedio)

    print(
        f"Generación {generacion + 1}/{GENERACIONES}: Mejor Precisión = {mejor_aptitud:.4f}, Mejor Cromosoma = {mejor_individuo_actual}, Precisión Promedio = {aptitud_promedio:.4f}")

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
indice_mejor_solucion = np.argmax([calcular_aptitud(ind) for ind in poblacion])
mejor_solucion = poblacion[indice_mejor_solucion]
indices_mejores_caracteristicas = np.where(mejor_solucion == 1)[0]
mejor_precision = calcular_aptitud(mejor_solucion)

print("\n--- Resultados Finales ---")
print(f"Mejor solución encontrada (cromosoma): {mejor_solucion}")
print(f"Precisión del modelo con estas características: {mejor_precision:.4f}")
print(
    f"Nombres de las características seleccionadas: {[iris.feature_names[i] for i in indices_mejores_caracteristicas]}")

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