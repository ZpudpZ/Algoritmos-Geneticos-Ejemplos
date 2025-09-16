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
OUTPUT_DIR = "output3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(FILE_PATH, header=0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    FEATURE_COUNT = X.shape[1]
    # Obtener los nombres de las características
    feature_names = X.columns.tolist()

    # Escalar los datos una sola vez
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Dataset cargado y escalado con {X.shape[0]} filas y {FEATURE_COUNT} características.\n")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta '{FILE_PATH}'. Asegúrate de que el archivo exista.")
    exit()

# --- 2. Configuración del Algoritmo Genético (con DEAP) ---

# a) Representación de la Población o Cromosomas
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Hiperparámetros a optimizar
C_VALUES = [0.01, 0.1, 1, 10, 100]
SOLVER_VALUES = ['lbfgs', 'liblinear']

# Atributos para la selección de características (57 bits)
for i in range(FEATURE_COUNT):
    toolbox.register(f"attr_feat_{i}", random.randint, 0, 1)

# Atributos para los hiperparámetros (2 bits/índices)
toolbox.register("attr_C", random.randint, 0, len(C_VALUES) - 1)
toolbox.register("attr_solver", random.randint, 0, len(SOLVER_VALUES) - 1)

# b) Inicialización
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (tuple(getattr(toolbox, f"attr_feat_{i}") for i in range(FEATURE_COUNT)) +
                  (toolbox.attr_C, toolbox.attr_solver)))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# --- 3. Función de Aptitud (Fitness) ---
def evaluate_individual(individual):
    """
    Evalúa la aptitud (fitness) de un individuo combinando
    la selección de características y los hiperparámetros para entrenar el modelo.
    """
    feature_selection_part = individual[:FEATURE_COUNT]
    hyperparameter_part = individual[FEATURE_COUNT:]

    selected_features_indices = [i for i, bit in enumerate(feature_selection_part) if bit == 1]

    if not selected_features_indices:
        return (0.0,)

    X_selected = X_scaled[:, selected_features_indices]

    C = C_VALUES[hyperparameter_part[0]]
    solver = SOLVER_VALUES[hyperparameter_part[1]]

    model = LogisticRegression(C=C, solver=solver, max_iter=5000)

    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return (accuracy,)


# --- 4. Operaciones Genéticas ---
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)


# --- 5. Ciclo de Vida del Algoritmo Genético ---
def main():
    random.seed(42)

    POPULATION_SIZE = 100
    GENERATIONS = 50
    CX_PROB = 0.7
    MUT_PROB = 0.5

    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox,
                                   cxpb=CX_PROB, mutpb=MUT_PROB,
                                   ngen=GENERATIONS,
                                   stats=stats,
                                   halloffame=hof, verbose=False)

    # --- 6. Guardar los resultados ---
    best_individual_raw = hof[0]
    best_accuracy = best_individual_raw.fitness.values[0]

    # Descomponer el mejor individuo en sus partes
    best_feature_selection = best_individual_raw[:FEATURE_COUNT]
    best_hyperparameters = best_individual_raw[FEATURE_COUNT:]

    selected_features_indices = [i for i, bit in enumerate(best_feature_selection) if bit == 1]
    selected_features_count = len(selected_features_indices)
    selected_names = [feature_names[i] for i in selected_features_indices]

    best_C = C_VALUES[best_hyperparameters[0]]
    best_solver = SOLVER_VALUES[best_hyperparameters[1]]

    print("\n--- Resultados del Algoritmo Genético Combinado ---")
    print(f"Mejor precisión (aptitud): {best_accuracy:.4f}")
    print(f"Número de características seleccionadas: {selected_features_count}")
    print("Mejor combinación de hiperparámetros:")
    print(f"  - C: {best_C}")
    print(f"  - solver: {best_solver}")

    # a. Guardar los resultados en un archivo de texto
    with open(os.path.join(OUTPUT_DIR, "resultados_combinados.txt"), "w") as f:
        f.write(f"Mejor precisión (aptitud): {best_accuracy:.4f}\n\n")
        f.write("Mejor combinación de hiperparámetros:\n")
        f.write(f"  - C: {best_C}\n")
        f.write(f"  - solver: {best_solver}\n\n")

        f.write(f"# Total de caracteristicas seleccionadas: {selected_features_count}\n\n")
        f.write("# Nombres de caracteristicas seleccionadas\n")
        f.write("\n".join(selected_names))
        f.write("\n\n")
        f.write("# Indices de caracteristicas seleccionadas\n")
        f.write(",".join(map(str, selected_features_indices)))

    # b. Generar gráfico de convergencia
    gen = log.select("gen")
    avg_fitness = log.select("avg")
    max_fitness = log.select("max")

    plt.figure(figsize=(8, 5))
    plt.plot(gen, avg_fitness, label="Aptitud Promedio")
    plt.plot(gen, max_fitness, label="Aptitud Máxima")
    plt.xlabel("Generación")
    plt.ylabel("Precisión")
    plt.title("Convergencia del Algoritmo Genético (Combinado)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "combinado_convergencia.png"))
    plt.close()

    # c. Generar el log en CSV
    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(OUTPUT_DIR, "combinado_log.csv"), index=False)

    print(f"\nResultados guardados en la carpeta '{OUTPUT_DIR}':")
    print("- resultados_combinados.txt")
    print("- combinado_convergencia.png")
    print("- combinado_log.csv")


if __name__ == "__main__":
    main()