# Algoritmos Genéticos

## 📌 Descripción del Proyecto  
Este repositorio contiene la implementación de **tres algoritmos genéticos** para la optimización de un modelo de **Regresión Logística**.  
El objetivo es **mejorar el rendimiento del modelo** para clasificar correos electrónicos como **spam o no spam**, utilizando el dataset **SpamBase**.  

El proyecto aborda la optimización en **tres fases**, cada una con un enfoque distinto para mostrar la versatilidad de los algoritmos genéticos:  

1. **Selección de Características**  
   - Un algoritmo genético identifica el subconjunto de características más relevantes del dataset, eliminando ruido y mejorando la eficiencia.  

2. **Optimización de Hiperparámetros**  
   - El algoritmo encuentra la mejor combinación de hiperparámetros (`C` y `solver`) para el modelo de Regresión Logística.  

3. **Neuroevolución**  
   - Aplica un AG para optimizar la arquitectura de una red neuronal, buscando la mejor configuración de neuronas y función de activación.**.  

---

## 📂 Estructura del Repositorio  
```bash
.
├── 1_Feature_Selection.py
├── 2_Hyperparameter_Optimization.py
├── 3_Neuroevolution.py
├── dataset/
│   └── spambase.csv
└── README.md
```
⚡ Las carpetas de salida (output1, output2, output3) se crean automáticamente al ejecutar los scripts y contienen resultados, gráficos y logs de cada experimento.

📊 Dataset
El dataset utilizado es SpamBase, un conjunto de datos de clasificación de correos electrónicos con 57 características que permiten predecir si un email es spam o no.

📎 Fuente: SpamBase Dataset en Kaggle ( https://www.kaggle.com/datasets/colormap/spambase )

⚙️ Requisitos

Asegúrate de tener Python 3 instalado. Luego, instala las dependencias con:
```bash
pip install pandas scikit-learn deap matplotlib numpy
```
▶️ Cómo Ejecutar

Ejecuta cada script desde la terminal:
```bash
python 1_Feature_Selection.py
python 2_Hyperparameter_Optimization.py
python 3_Neuroevolution.py
```
📑 Resultados y Archivos Generados
  
  📁 output1/ (Selección de Características)

    caracteristicas_seleccionadas.txt → Precisión, cantidad y nombres de las características óptimas.

    ga_convergencia.png → Gráfico de la evolución de la precisión.

  📁 output2/ (Optimización de Hiperparámetros)

    mejores_hiperparametros.txt → Precisión y mejor combinación de C y solver.

    hiperparametros_convergencia.png → Gráfico de la evolución de la precisión.

  📁 output3/ (Neuroevolución)

      mejor_arquitectura.txt: Archivo de texto que detalla la precisión y la mejor arquitectura de la red neuronal (número de neuronas y función de activación).

      evolucion_aptitud.png: Gráfico que muestra el progreso de la aptitud máxima y promedio de la población de redes neuronales.

      evolucion_log.csv: Un registro en CSV de la evolución del proceso de neuroevolución.
