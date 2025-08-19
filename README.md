# Clasificación de Imágenes de Mariposas con MLP y CNN

Este repositorio contiene la implementación de un proyecto de clasificación de imágenes de mariposas utilizando redes neuronales de tipo Perceptrón Multicapa (MLP) y Red Neuronal Convolucional (CNN) en PyTorch, según las especificaciones de las instrucciones proporcionadas. El proyecto utiliza el dataset "Leeds Butterfly Dataset" para clasificar 10 especies de mariposas, comparando el rendimiento de las arquitecturas MLP y CNN.

## Descripción del Proyecto
- **Dataset**: Leeds Butterfly Dataset (832 imágenes, 10 clases) disponible en Kaggle.
- **Tarea**: Clasificación multiclase de imágenes de mariposas usando MLP y CNN.
- **Preprocesamiento**: Imágenes redimensionadas a 64x64, normalizadas (media/desviación estándar de ImageNet), con aumento de datos ligero (rotaciones, volteos horizontales) para entrenamiento.
- **Modelos**:
  - **MLP**: Configurable con diferentes capas, neuronas, funciones de activación (ReLU, Tanh, Sigmoid) y Dropout (0.5).
  - **CNN**: Arquitectura simple con dos capas convolucionales, BatchNorm, ReLU, MaxPooling y capas densas con Dropout.
- **Evaluación**: Métricas incluyen precisión (accuracy), F1 macro, F1 ponderado, matrices de confusión y curvas de pérdida/precisión.
- **Comparación**: Análisis comparativo del rendimiento de MLP y CNN, incluyendo tiempos de entrenamiento y análisis del sesgo inductivo.

## Requisitos
- Python 3.8+
- Google Colab con GPU (recomendado) o entorno local con soporte CUDA
- Google Drive para almacenamiento del dataset

Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Instrucciones de Configuración
1. **Clonar el Repositorio**:
   ```bash
   git clone <url-del-repositorio>
   cd <carpeta-del-repositorio>
   ```

2. **Descargar el Dataset**:
   - Descarga el Leeds Butterfly Dataset desde [Kaggle](https://www.kaggle.com/datasets/veeralakrishna/butterfly-dataset).
   - Descomprime y organiza el dataset en Google Drive con la siguiente estructura:
     ```
     /content/drive/MyDrive/UTEC/CD & IA/Ciclo III/Machine Learning/Entrenamientos/Clasificación/Redes Neuronales/Análisis de Imágenes/Dataset/
     ├── 001. Danaus plexippus/
     ├── 002. Heliconius charitonius/
     ...
     ├── 010. Vanessa cardui/
     ```
   - Asegúrate de que la variable `DATASET_PATH` en `train_evaluate.py` coincida con la ruta de tu Google Drive.

3. **Configurar el Entorno**:
   - Instala las dependencias desde `requirements.txt`.
   - Si usas Google Colab, sube `train_evaluate.py` y `requirements.txt` al entorno de Colab.

4. **Ejecutar el Script**:
   ```bash
   python train_evaluate.py
   ```
   - El script realizará:
     - Carga y preprocesamiento del dataset (división 80/10/10).
     - Entrenamiento y evaluación de cuatro configuraciones de MLP y una CNN.
     - Guardado de resultados (métricas, matrices de confusión, curvas) en la carpeta `results/`.

## Resultados
- **Archivos de Salida** (en `results/`):
  - `MLP_Config_X_cm_test.png`: Matriz de confusión para cada configuración de MLP.
  - `MLP_Config_X_curves.png`: Curvas de pérdida y precisión para cada configuración de MLP.
  - `CNN_cm_test.png`: Matriz de confusión para la CNN.
  - `CNN_curves.png`: Curvas de pérdida y precisión para la CNN.
  - `mlp_vs_cnn_curves.png`: Curvas comparativas del mejor MLP y la CNN.
  - `comparison.txt`: Tabla comparativa de métricas entre el mejor MLP y la CNN.
  - `confusion_matrices.txt`: Archivo de texto con las matrices de confusión.
- **Hallazgos Clave**:
  - La CNN supera al MLP (precisión: 61.45% vs. 36.14%, F1 macro: 60.12% vs. 34.78%).
  - La CNN es más eficiente (~4.8s vs. ~5.2s por época) y adecuada para imágenes debido a su sesgo inductivo espacial.
  - Clases difíciles (por ejemplo, Heliconius erato, Junonia coenia) destacan las limitaciones del dataset (tamaño pequeño, desbalance).

## Trabajo Futuro
- Aumentar el número de épocas (20-50) para mejorar la convergencia.
- Usar imágenes de mayor tamaño (por ejemplo, 224x224) para la CNN.
- Aplicar aumento de datos avanzado (RandomCrop, ColorJitter).
- Balancear clases con pérdida ponderada o sobremuestreo.
- Explorar aprendizaje por transferencia con modelos preentrenados (por ejemplo, ResNet18).

## Licencia
Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

## Agradecimientos
- Leeds Butterfly Dataset: [Kaggle](https://www.kaggle.com/datasets/veeralakrishna/butterfly-dataset)
- PyTorch y bibliotecas relacionadas para la implementación de deep learning.