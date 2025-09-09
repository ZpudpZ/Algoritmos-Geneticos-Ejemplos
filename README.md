# Algoritmos-Geneticos-Ejemplos
Este repositorio contiene implementaciones en Python de algoritmos genéticos. Incluye ejemplos de Feature Selection, Hyperparameter Optimization y Neuroevolution.
# Ejemplos de Algoritmos Genéticos en Python

## 📜 Contenido

* [1. Selección de Características](#1-selección-de-características)
* [2. Optimización de Hiperparámetros](#2-optimización-de-hiperparámetros)
* [3. Neuroevolución](#3-neuroevolución)
* [4. Requisitos](#4-requisitos)

## 1. Selección de Características

-   **Archivo**: `1_Feature Selection.py`
-   **Descripción**: Este script utiliza un algoritmo genético para identificar el subconjunto de características más relevante del conjunto de datos de Iris. El objetivo es maximizar la **precisión** de un modelo de Regresión Logística, demostrando cómo el AG puede simplificar modelos y mejorar su rendimiento al eliminar datos irrelevantes.
-   **Optimización**: El AG busca la mejor combinación de `0`s y `1`s, donde cada bit representa si una característica es seleccionada o no.

---
## 2. Optimización de Hiperparámetros

-   **Archivo**: `2_Hyperparameter Optimization.py`
-   **Descripción**: Este ejemplo se centra en el ajuste fino de un clasificador K-Vecinos Más Cercanos (K-NN). El algoritmo genético optimiza los valores de los hiperparámetros `n_neighbors` (número de vecinos) y `p` (tipo de distancia) para lograr la mayor **precisión** del modelo.
-   **Optimización**: El cromosoma del AG representa una combinación de hiperparámetros. El `fitness` se mide por la precisión del modelo K-NN con esa combinación.

---
## 3. Neuroevolución

-   **Archivo**: `3_Neuroevolution.py`
-   **Descripción**: Un ejemplo de cómo los algoritmos genéticos pueden ser usados para entrenar una red neuronal. El AG optimiza directamente los **pesos y sesgos** de una pequeña red neuronal diseñada para resolver la función lógica **XOR**. A diferencia de los métodos tradicionales, la red "aprende" a través de la selección y mutación, en lugar de un proceso de *backpropagation*.
-   **Optimización**: El cromosoma es un array de números flotantes (los pesos y sesgos de la red), y la aptitud se calcula en función de qué tan bien la red minimiza el error de predicción.

## 4. Requisitos

Para ejecutar los ejemplos, asegúrate de tener Python instalado junto con las siguientes librerías. Puedes instalarlas con `pip`:

```bash
pip install numpy scikit-learn matplotlib
```
