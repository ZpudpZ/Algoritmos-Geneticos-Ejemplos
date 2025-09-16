import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib

# --- 1. Cargar y preparar los datos ---
FILE_PATH = "dataset/spambase.csv"
OUTPUT_DIR = "output1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    df = pd.read_csv(FILE_PATH, header=0)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    NUM_FEATURES = X.shape[1]
    feature_names = X.columns.tolist()
    print(f"Dataset cargado con {X.shape[0]} filas y {NUM_FEATURES} características.\n")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta '{FILE_PATH}'. Asegúrate de que el archivo exista.")
    exit()

# --- 2. Configuración del Algoritmo Genético (con DEAP) ---

# a) Representación de la Población o Cromosomas
# Definimos la "aptitud" (Fitness) que el algoritmo busca maximizar.
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# Definimos el "cromosoma" o "individuo" como una lista de bits.
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# Cada "gen" del cromosoma es un bit (0 o 1).
toolbox.register("attr_bool", random.randint, 0, 1)

# b) Inicialización
# Un individuo es una lista de 57 genes, uno para cada característica.
# 1 significa 'seleccionar', 0 significa 'no seleccionar'.
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=NUM_FEATURES)
# La población es una lista de individuos.
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# --- 3. Función de Aptitud (Fitness) ---
def evaluate_individual(individual):
    """
    Calcula la aptitud (fitness) de un individuo.
    Se entrena un modelo de regresión logística con las características seleccionadas
    y se devuelve la precisión como métrica de rendimiento.
    """
    if sum(individual) == 0:
        return (0.0,)  # Si no se selecciona ninguna característica, la aptitud es cero.

    selected_features_indices = [i for i, bit in enumerate(individual) if bit == 1]
    X_selected = X.iloc[:, selected_features_indices]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return (accuracy,)


# --- 4. Operaciones Genéticas ---
toolbox.register("evaluate", evaluate_individual)
# c) Selección: Elige los individuos para reproducirse (cruce y mutación).
toolbox.register("select", tools.selTournament, tournsize=3)
# d) Cruzamiento: Combina los genes de dos individuos.
toolbox.register("mate", tools.cxTwoPoint)
# e) Mutación: Cambia aleatoriamente los genes de un individuo.
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)


# --- 5. Ciclo de Vida del Algoritmo Genético ---
def main():
    random.seed(42)

    POPULATION_SIZE = 100
    GENERATIONS = 50
    CX_PROB = 0.7  # Probabilidad de cruce
    MUT_PROB = 0.2  # Probabilidad de mutación

    # Inicialización de la población
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
    best_individual = hof[0]
    best_accuracy = best_individual.fitness.values[0]
    selected_features_indices = [i for i, bit in enumerate(best_individual) if bit == 1]

    selected_features_count = len(selected_features_indices)
    selected_names = [feature_names[i] for i in selected_features_indices]

    print("\n--- Resultados del Algoritmo Genético ---")
    print(f"Mejor precisión (aptitud): {best_accuracy:.4f}")
    print(f"Número de características seleccionadas: {selected_features_count}")
    print("El cromosoma óptimo es:", best_individual)

    # a. Generar archivo de texto de características
    with open(os.path.join(OUTPUT_DIR, "caracteristicas_seleccionadas.txt"), "w") as f:
        f.write(f"Mejor precisión (aptitud): {best_accuracy:.4f}\n\n")  # <-- Se agrega esta línea
        f.write(f"# Total de caracteristicas seleccionadas: {selected_features_count}\n\n")
        f.write("# Índices de features seleccionadas\n")
        f.write(",".join(map(str, selected_features_indices)) + "\n\n")
        f.write("# Nombres de features\n")
        f.write("\n".join(selected_names))

    # b. Generar gráfico de convergencia
    gen = log.select("gen")
    avg_fitness = log.select("avg")
    max_fitness = log.select("max")

    plt.figure(figsize=(8, 5))
    plt.plot(gen, avg_fitness, label="Aptitud Promedio")
    plt.plot(gen, max_fitness, label="Aptitud Máxima")
    plt.xlabel("Generación")
    plt.ylabel("Precisión")
    plt.title("Convergencia del Algoritmo Genético")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "ga_convergencia.png"))
    plt.close()

    # c. Generar ga_log.csv
    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(OUTPUT_DIR, "ga_log.csv"), index=False)

    print(f"\nResultados guardados en la carpeta '{OUTPUT_DIR}':")
    print("- caracteristicas_seleccionadas.txt")
    print("- ga_convergencia.png")
    print("- ga_log.csv")


if __name__ == "__main__":
    main()