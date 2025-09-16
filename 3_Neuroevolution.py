import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from deap import base, creator, tools, algorithms

# --- 1. Configuración y Rutas ---
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_PATH = "dataset/spambase.csv"
OUT_DIR = "output3"
os.makedirs(OUT_DIR, exist_ok=True)


# --- 2. Carga y Preprocesamiento de Datos ---
def load_and_preprocess_data():
    """Carga y prepara el dataset spambase, asegurando que todos los datos sean numéricos."""
    try:
        # Carga el CSV sin encabezado, ya que el archivo puede contener texto incrustado
        df = pd.read_csv(DATA_PATH, header=None)

        # La solución robusta: convertir todas las columnas a numérico.
        # Los valores no numéricos se convertirán en NaN.
        df_numeric = df.apply(pd.to_numeric, errors='coerce')

        # Eliminar cualquier fila que contenga un valor NaN (es decir, una fila con texto)
        df_clean = df_numeric.dropna(axis=0, how='any').reset_index(drop=True)

        # Separar las características (X) y las etiquetas (y)
        X = df_clean.iloc[:, :-1]
        y = df_clean.iloc[:, -1]

        # Escalar las características
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        print(
            f"[INFO] Dataset cargado y preprocesado con {X_scaled.shape[0]} filas y {X_scaled.shape[1]} características.")

        return X_scaled, y

    except FileNotFoundError:
        print(f"[ERROR] No se encontró el archivo en la ruta '{DATA_PATH}'. Asegúrate de que el archivo exista.")
        exit()


# Cargar y dividir los datos una sola vez
X, y = load_and_preprocess_data()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

# --- 3. Representación del Individuo (Cromosoma) ---
# Opciones para la arquitectura de la red neuronal
NEURONS_OPTIONS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
ACTIVATION_OPTIONS = ['relu', 'logistic', 'tanh']

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_neurons", random.choice, NEURONS_OPTIONS)
toolbox.register("attr_activation", random.choice, ACTIVATION_OPTIONS)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_neurons, toolbox.attr_activation))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# --- 4. Función de Aptitud (Fitness) ---
def eval_individual(individual):
    """Evalúa un individuo entrenando una MLP y devuelve su precisión."""
    neurons = individual[0]
    activation = individual[1]

    clf = MLPClassifier(hidden_layer_sizes=(neurons,),
                        activation=activation,
                        solver='adam',
                        max_iter=300,
                        random_state=RANDOM_SEED,
                        early_stopping=True)

    try:
        clf.fit(X_train, y_train)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
    except Exception as e:
        acc = 0.0
        print(f"[WARN] Error al evaluar individuo {individual}: {e}")

    return (acc,)


toolbox.register("evaluate", eval_individual)

# --- 5. Operadores Genéticos ---
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxUniform, indpb=0.5)


def mutate_individual(individual):
    """Mutación: cambia neuronas o activación con cierta probabilidad."""
    if random.random() < 0.5:
        individual[0] = random.choice(NEURONS_OPTIONS)
    if random.random() < 0.5:
        individual[1] = random.choice(ACTIVATION_OPTIONS)
    return individual,


toolbox.register("mutate", mutate_individual)


# --- 6. Ejecución del Algoritmo Genético ---
def main():
    POP_SIZE = 50
    CX_PB = 0.7
    MUT_PB = 0.2
    NGEN = 50

    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print("[INFO] Iniciando evolución...")
    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CX_PB, mutpb=MUT_PB,
                                   ngen=NGEN, stats=stats, halloffame=hof,
                                   verbose=False)

    # --- 7. Guardar Resultados ---
    best_ind = hof[0]
    best_neurons = best_ind[0]
    best_activation = best_ind[1]
    best_fitness = best_ind.fitness.values[0]

    # Guardar log en CSV
    log_df = pd.DataFrame(log)
    log_df.to_csv(os.path.join(OUT_DIR, "evolucion_log.csv"), index=False)

    # Graficar la evolución
    gens = log_df['gen']
    max_fitness = log_df['max']
    avg_fitness = log_df['avg']
    plt.figure(figsize=(8, 5))
    plt.plot(gens, max_fitness, label="Aptitud Máxima")
    plt.plot(gens, avg_fitness, label="Aptitud Promedio")
    plt.title("Evolución de la Aptitud")
    plt.xlabel("Generación")
    plt.ylabel("Precisión")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUT_DIR, "evolucion_aptitud.png"))
    plt.close()

    # Guardar el mejor resultado
    with open(os.path.join(OUT_DIR, "mejor_arquitectura.txt"), "w") as f:
        f.write("--- Mejor Arquitectura Encontrada ---\n")
        f.write(f"Neuronas: {best_neurons}\n")
        f.write(f"Activación: {best_activation}\n")
        f.write(f"Precisión de Validación: {best_fitness:.4f}\n")

    print("\n[INFO] Proceso de neuroevolución completado. Resultados en la carpeta 'output3'.")
    print(f"Mejor arquitectura: {best_ind} con precisión: {best_fitness:.4f}")


if __name__ == "__main__":
    main()