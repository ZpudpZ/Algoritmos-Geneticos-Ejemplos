import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib

# --- 1. Cargar y preparar los datos ---
FILE_PATH = "dataset/spambase.csv"
OUTPUT_DIR = "output2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(FILE_PATH, header=0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Escalar los datos una sola vez
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dataset cargado y escalado con {X.shape[0]} filas y {X.shape[1]} características.\n")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta '{FILE_PATH}'. Asegúrate de que el archivo exista.")
    exit()

# --- 2. Configuración del Algoritmo Genético (con DEAP) ---

# a) Representación de la Población o Cromosomas
# Definimos la "aptitud" (Fitness) que el algoritmo busca maximizar.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Un individuo ahora es una lista de 2 elementos: [índice_C, índice_solver].
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Hiperparámetro 1: C (inversa de la fuerza de regularización). Usamos un mapeo.
# Valores posibles para C: 0.01, 0.1, 1, 10, 100
C_VALUES = [0.01, 0.1, 1, 10, 100]
# Hiperparámetro 2: solver (algoritmo para optimización). Usamos un mapeo.
# Valores posibles para solver: 'lbfgs', 'liblinear'
SOLVER_VALUES = ['lbfgs', 'liblinear']

# Atributos: un índice para cada hiperparámetro.
toolbox.register("attr_C", random.randint, 0, len(C_VALUES) - 1)
toolbox.register("attr_solver", random.randint, 0, len(SOLVER_VALUES) - 1)

# b) Inicialización
# Un individuo se crea con 2 atributos.
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_C, toolbox.attr_solver))
# La población es una lista de individuos.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# --- 3. Función de Aptitud (Fitness) ---
def evaluate_individual(individual):
    """
    Evalúa a un individuo (una combinación de hiperparámetros)
    entrenando un modelo con todos los datos.
    """
    # Mapear los índices del cromosoma a los valores reales de los hiperparámetros.
    C = C_VALUES[individual[0]]
    solver = SOLVER_VALUES[individual[1]]

    # Crear y entrenar el modelo con los hiperparámetros seleccionados.
    model = LogisticRegression(C=C, solver=solver, max_iter=5000)

    # Dividir y entrenar el modelo.
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    # Devolver la precisión como aptitud (fitness).
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return (accuracy,)


# --- 4. Operaciones Genéticas ---
toolbox.register("evaluate", evaluate_individual)
# c) Selección: Elige los individuos para reproducirse.
toolbox.register("select", tools.selTournament, tournsize=3)
# d) Cruzamiento: Combina los hiperparámetros de dos padres.
toolbox.register("mate", tools.cxUniform, indpb=0.5)

# e) Mutación: Cambia un hiperparámetro por otro aleatorio.
def mutate_individual(individual):
    if random.random() < 0.5:
        individual[0] = random.randint(0, len(C_VALUES) - 1)
    if random.random() < 0.5:
        individual[1] = random.randint(0, len(SOLVER_VALUES) - 1)
    return individual,

toolbox.register("mutate", mutate_individual)


# --- 5. Ciclo de Vida del Algoritmo Genético ---
def main():
    random.seed(42)

    POPULATION_SIZE = 100
    GENERATIONS = 50
    CX_PROB = 0.7
    MUT_PROB = 0.5

    # Inicialización de la población.
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # f) Ciclo de Terminación: Se repite por un número de generaciones.
    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=CX_PROB, mutpb=MUT_PROB,
                                   ngen=GENERATIONS,
                                   stats=stats,
                                   halloffame=hof, verbose=False)

    # --- 6. Guardar los resultados ---
    best_individual_raw = hof[0]
    best_accuracy = best_individual_raw.fitness.values[0]

    # Mapear el mejor individuo a los valores reales de los hiperparámetros.
    best_C = C_VALUES[best_individual_raw[0]]
    best_solver = SOLVER_VALUES[best_individual_raw[1]]

    print("\n--- Resultados del Algoritmo Genético ---")
    print(f"Mejor precisión (aptitud): {best_accuracy:.4f}")
    print(f"Mejor combinación de hiperparámetros:")
    print(f"  - C: {best_C}")
    print(f"  - solver: {best_solver}")

    # a. Guardar los resultados en un archivo de texto.
    with open(os.path.join(OUTPUT_DIR, "mejores_hiperparametros.txt"), "w") as f:
        f.write(f"Mejor precisión (aptitud): {best_accuracy:.4f}\n\n")
        f.write("Mejor combinación de hiperparámetros:\n")
        f.write(f"  - C: {best_C}\n")
        f.write(f"  - solver: {best_solver}\n")

    # b. Generar gráfico de convergencia.
    gen = log.select("gen")
    avg_fitness = log.select("avg")
    max_fitness = log.select("max")

    plt.figure(figsize=(8, 5))
    plt.plot(gen, avg_fitness, label="Aptitud Promedio")
    plt.plot(gen, max_fitness, label="Aptitud Máxima")
    plt.xlabel("Generación")
    plt.ylabel("Precisión")
    plt.title("Convergencia del Algoritmo Genético (Hiperparámetros)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "hiperparametros_convergencia.png"))
    plt.close()

    # c. Generar el log en CSV.
    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(OUTPUT_DIR, "hiperparametros_log.csv"), index=False)

    print(f"\nResultados guardados en la carpeta '{OUTPUT_DIR}':")
    print("- mejores_hiperparametros.txt")
    print("- hiperparametros_convergencia.png")
    print("- hiperparametros_log.csv")


if __name__ == "__main__":
    main()