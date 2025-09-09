#Ejemplo para Neuroevolución (Neuroevolution)
import numpy as np
import matplotlib.pyplot as plt

# 1. Parámetros del algoritmo genético
TAMANO_POBLACION = 50
GENERACIONES = 100
TASA_MUTACION = 0.05
LONGITUD_CROMOSOMA = 9  # (2*2 + 2) + (2*1 + 1) = 4 + 2 + 2 + 1 = 9 pesos y sesgos
# Pesos: w11, w12, w21, w22 (4)
# Sesgos: b1, b2 (2)
# Pesos capa oculta a salida: w31, w32 (2)
# Sesgo de salida: b3 (1)

# Datos de la función XOR
entradas_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
salidas_xor = np.array([[0], [1], [1], [0]])


def crear_individuo():
    """Crea un individuo (cromosoma) con pesos y sesgos aleatorios."""
    return np.random.uniform(-1, 1, size=LONGITUD_CROMOSOMA)


def sigmoid(x):
    """Función de activación Sigmoid."""
    return 1 / (1 + np.exp(-x))


def calcular_aptitud(individuo):
    """Evalúa la aptitud del individuo (precisión de la red neuronal)."""
    # Desempaquetar los pesos y sesgos del cromosoma
    w_oculta = individuo[0:4].reshape(2, 2)
    b_oculta = individuo[4:6].reshape(1, 2)
    w_salida = individuo[6:8].reshape(2, 1)
    b_salida = individuo[8]

    # Propagación hacia adelante
    capa_oculta_activacion = sigmoid(np.dot(entradas_xor, w_oculta) + b_oculta)
    salida_predicha = sigmoid(np.dot(capa_oculta_activacion, w_salida) + b_salida)

    # Calcular la diferencia (error)
    error = np.sum(np.abs(salida_predicha - salidas_xor))

    # La aptitud es inversamente proporcional al error
    return 1 / (1 + error)


def seleccionar_padres(poblacion, puntajes_aptitud):
    """Selecciona dos padres basados en la ruleta de selección."""
    total_aptitud = sum(puntajes_aptitud)
    if total_aptitud == 0:
        total_aptitud = 1

    probabilidades = [score / total_aptitud for score in puntajes_aptitud]
    parent_indices = np.random.choice(range(TAMANO_POBLACION), size=2, p=probabilidades, replace=False)

    return poblacion[parent_indices[0]], poblacion[parent_indices[1]]


def cruzar(padre1, padre2):
    """Cruce de un solo punto."""
    crossover_point = np.random.randint(1, LONGITUD_CROMOSOMA - 1)
    hijo1 = np.concatenate((padre1[:crossover_point], padre2[crossover_point:]))
    hijo2 = np.concatenate((padre2[:crossover_point], padre1[crossover_point:]))
    return hijo1, hijo2


def mutar(individual):
    """Mutación aleatoria."""
    for i in range(LONGITUD_CROMOSOMA):
        if np.random.rand() < TASA_MUTACION:
            # Añade un pequeño valor aleatorio al peso o sesgo
            individual[i] += np.random.uniform(-0.5, 0.5)
    return individual


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

    # Imprimimos el progreso
    print(
        f"Generación {generacion + 1}/{GENERACIONES}: Mejor Aptitud = {mejor_aptitud:.4f}, Precisión Promedio = {aptitud_promedio:.4f}")

    nueva_poblacion = []

    # Elitismo: mantenemos al mejor individuo
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
mejor_aptitud_final = mejor_aptitud_por_generacion[-1]

print("\n--- Resultados Finales ---")
print(f"Mejor combinación de pesos y sesgos (cromosoma): {mejor_solucion_final}")
print(f"Aptitud final (inversamente proporcional al error): {mejor_aptitud_final:.4f}")

# Gráfico de la evolución de la aptitud
plt.figure(figsize=(10, 6))
plt.plot(range(GENERACIONES), mejor_aptitud_por_generacion, label='Mejor Aptitud por Generación', color='blue',
         marker='o', markersize=4)
plt.plot(range(GENERACIONES), aptitud_promedio_por_generacion, label='Aptitud Promedio por Generación', color='orange',
         linestyle='--')
plt.title('Evolución de la Aptitud de la Red Neuronal')
plt.xlabel('Generación')
plt.ylabel('Aptitud')
plt.grid(True)
plt.legend()
plt.show()