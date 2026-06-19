# Clasificación de Gestos de la LSM mediante sEMG y Redes Neuronales Artificiales

**Universidad Veracruzana - Programa de Ingeniería Biomédica** **Proyecto Integrador (Monografía)**

Este repositorio contiene el código fuente, la metodología y los resultados de un sistema computacional diseñado para clasificar tres gestos estáticos (Letras A, B y C) de la Lengua de Señas Mexicana (LSM). El sistema decodifica biopotenciales musculares (sEMG) del antebrazo utilizando instrumentación clínica y algoritmos de aprendizaje profundo.

## ⚙️ Arquitectura y Flujo de Procesamiento (*Pipeline*)

El procesamiento de las bioseñales se estructuró bajo un flujo de trabajo de extremo a extremo (*End-to-End*) adaptado para instrumentación clínica:

1. **Adquisición de Datos:** Se empleó un sistema BIOPAC MP36 con una frecuencia de muestreo de 2000 Hz. La base de datos final consta de registros de 13 sujetos distintos, siguiendo recomendaciones SENIAM.
2. **Preprocesamiento:** Las señales crudas se acondicionan mediante un filtro digital pasabanda Butterworth de 4.° orden (20 - 450 Hz) para atenuar ruido e interferencias, seguido de una rectificación por valor absoluto.
3. **Extracción de Características:** La señal se segmenta en ventanas de 6000 ms. Se extraen descriptores temporales dominantes:
   * Valor Cuadrático Medio (RMS)
   * Valor Absoluto Medio (MAV)
   * Varianza (VAR)
   * Longitud de Onda (WL)
4. **Clasificación (ANN):** Red Neuronal Artificial Multicapa con capas ocultas de 32 y 16 neuronas, optimizada con regularización *Dropout* (0.2) y *Batch Normalization* para prevenir el sobreajuste.

## 📊 Resultados Principales

Para evaluar el rigor metodológico del sistema, se estructuraron dos escenarios de validación distintos.

### 1. Evaluación Intrasujeto (Potencial del Modelo)
Se aplicó una división aleatoria de los datos para evaluar la capacidad de la red cuando conoce la anatomía basal de los usuarios. 
* **Accuracy Global:** **75.00%**
* El modelo demostró su mejor desempeño en la clasificación de la **Seña A**, alcanzando un *F1-score* de 0.88.

> ![Matriz de Confusión Intrasujeto](ruta/a/tu/imagen_matriz_75.png)  
> *Figura 1. Matriz de confusión evidenciando una separación efectiva, con zonas de reto concentradas entre las señas B y C.*

### 2. Evaluación Intersujeto (Validación LOSO)
Para cumplir con los criterios de generalización del mundo real y simular la llegada de usuarios nuevos, se implementó una validación cruzada *Leave-One-Subject-Out* (LOSO).
* **Accuracy Promedio:** **51.91%**
* **Degradación del Rendimiento:** **23.09%**

> ![Resultados LOSO](ruta/a/tu/imagen_matriz_loso.png)  
> *Figura 2. Matriz de confusión de la evaluación LOSO.*

**Análisis Clínico:** La degradación del 23.09% supera la tolerancia de diseño (15%). Este fenómeno evidencia la alta dependencia del clasificador a la variabilidad anatómica inter-sujeto. El error se concentra en la clasificación errónea de la Seña B como Seña C, justificado biomecánicamente por el solapamiento en el reclutamiento isométrico del músculo flexor profundo de los dedos.

## 🚀 Conclusiones y Trabajo Futuro

El modelo base cumple exitosamente con el objetivo de clasificación *offline*, demostrando la viabilidad de la extracción de descriptores temporales a partir del BIOPAC MP36. No obstante, los resultados de la evaluación LOSO concluyen que un enfoque de clasificación universal es matemáticamente insuficiente. 

Para escalar este desarrollo hacia aplicaciones embebidas o prótesis en tiempo real, se establece como trabajo futuro la integración de un módulo de calibración dinámica (*Transfer Learning*) que adapte los pesos de la red a la firma fisiológica específica de cada nuevo paciente.

## 📂 Estructura del Repositorio

* `pipeline_loso_end_to_end.py`: Script principal en Python que contiene el flujo completo de lectura de archivos, preprocesamiento, construcción de la ANN y la validación cruzada LOSO.
* `/datos`: Directorio destinado a alojar los registros `.txt` exportados del sistema de adquisición.
