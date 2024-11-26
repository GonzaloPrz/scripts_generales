# Repositorio de Scripts Generales

Este repositorio contiene una colección de scripts en Python diseñados para diversas tareas de procesamiento de datos, análisis estadístico y aprendizaje automático. A continuación, se presenta una descripción general de los scripts, sus propósitos y sus funciones principales.

## Estructura del Repositorio

- **`utils.py`**: Contiene utilidades de propósito general para el aprendizaje automático, incluyendo eliminación recursiva de características, ajuste de hiperparámetros y comparación de modelos.
- **`stat_analysis.py`**: Se centra en análisis estadístico, estadísticos descriptivos y visualización de datos.
- **`matching_module.py`**: Implementa estimación de puntuaciones de propensión y emparejamiento de vecinos más cercanos para estudios de inferencia causal.

---

## Descripción de los Scripts

### 1. `utils.py`
Este script proporciona funciones utilitarias para tareas de aprendizaje automático y evaluación de modelos. Funcionalidades clave:

- **Eliminación Recursiva de Características (`recursive_feature_elimination`)**:
  - Selecciona el mejor subconjunto de características para un modelo dado mediante un proceso recursivo.
  - Compatible con tareas de clasificación y regresión.
  - Métricas soportadas incluyen AUC, entropía cruzada y métricas basadas en costos.

- **Ajuste de Hiperparámetros (`tuning`)**:
  - Optimización bayesiana para encontrar los mejores hiperparámetros.
  - Métricas de evaluación personalizables y compatibles con clasificación y regresión.

- **Comparación de Modelos (`compare`)**:
  - Evalúa y compara múltiples modelos utilizando técnicas de remuestreo (bootstrap).

### 2. `stat_analysis.py`
Este script está diseñado para análisis estadístico y visualización. Funcionalidades clave:

- **Pruebas Estadísticas (`analyze_data`)**:
  - Selecciona y ejecuta automáticamente las pruebas estadísticas adecuadas según las características de los datos.
  - Soporta pruebas T, ANOVA, Mann-Whitney U y Kruskal-Wallis.

- **Estadísticas Descriptivas (`stat_describe`)**:
  - Calcula estadísticas resumidas para una característica o datos agrupados.
  - Proporciona la salida en formato tabular.

- **Gráficos Descriptivos (`descriptive_plots`)**:
  - Genera histogramas, gráficos QQ y gráficos de violín para visualizar distribuciones de datos.

### 3. `matching_module.py`
Este script se centra en la inferencia causal a través del emparejamiento basado en puntuaciones de propensión. Funcionalidades clave:

- **Estimación de Puntuaciones de Propensión (`estimate_propensity_scores`)**:
  - Calcula puntuaciones de propensión utilizando regresión logística.

- **Emparejamiento de Vecinos Más Cercanos (`perform_matching`)**:
  - Empareja grupos tratados y de control en función de las puntuaciones de propensión.
  - Soporta emparejamiento basado en calipers para controlar el desequilibrio entre grupos.

---

## Comenzando

### Requisitos Previos
- Python 3.8+

Instala las dependencias mediante:
```bash
pip install -r requirements.txt
