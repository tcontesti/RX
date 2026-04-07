# spark/

Código que corre sobre la NVIDIA DGX Spark (GPU Grace Hopper GB10, arquitectura aarch64). Tres bloques:

```
spark/
├── worker/      servicio de inferencia en línea (RabbitMQ + GPU)
├── scripts/     entrenamiento, evaluación y preprocesado
└── pipeline/    pipeline reutilizable (data, models, training, evaluation)
```

Los pesos entrenados y los datasets no se versionan en este repositorio. Las rutas absolutas que aparecen en los YAML y los servicios (`/home/$SPARK_USER/nodule_detection/...`) reflejan el despliegue en el servidor del hospital y deben adaptarse a cada entorno.

## worker/

Servicio que escucha la cola `cxr.inference` de RabbitMQ, ejecuta el ensemble Faster R-CNN + YOLOv8 + WBF sobre cada radiografía y publica el resultado en `cxr.results`.

Archivos:

- `inference_worker.py` — entrada principal. Carga modelos al arrancar, mantiene la sesión CUDA caliente, valida cada imagen (10×10 a 16384×16384 píxeles) y aplica reconexión con backoff exponencial si RabbitMQ se cae.
- `config.yaml` — configuración del worker. Las credenciales y la IP del broker se inyectan como placeholders (`${RABBITMQ_HOST}`, `${RABBITMQ_PASS}`); deben sustituirse antes de arrancar.
- `cxr-worker.service` — unidad systemd para auto-arranque tras reinicio.
- `start_worker.sh` — verifica disponibilidad de GPU y activa el venv antes de lanzar el worker.

Para arrancar manualmente:

```bash
cd ~/nodule_detection
./worker/start_worker.sh
```

Para instalar como servicio (en la Spark):

```bash
sudo cp worker/cxr-worker.service /etc/systemd/system/
sudo systemctl enable cxr-worker
sudo systemctl start cxr-worker
```

## scripts/

Scripts puntuales para entrenar, evaluar o preprocesar. No comparten estado entre ejecuciones — cada uno es un punto de entrada independiente.

| Script | Qué hace |
|---|---|
| `preprocess_node21.py` | Convierte el dataset NODE21 (`.mha`) a PNG agrupados por estudio |
| `generate_augmented_images.py` | Genera augmentaciones offline manteniendo agrupación por imagen (evita el data leakage del trabajo de referencia) |
| `train_frcnn_vindr.py` | Entrena Faster R-CNN con backbone ResNet-50 preentrenado sobre VinDr-CXR |
| `train_frcnn_reproduce.py` | Reproduce exactamente la configuración Behrendt et al. para baseline comparativo |
| `train_yolo.py` | Entrena YOLOv8s o YOLOv5x sobre NODE21 a resolución 1024×1024 |
| `evaluate.py` | Evaluación FROC + métricas NODE21 (Sens@0.25FP, CM) y AUROC |
| `evaluate_original_frcnn.py` | Evalúa el checkpoint Behrendt original sobre el split limpio |
| `ensemble_wbf.py` | Fusión Weighted Box Fusion de las predicciones de FRCNN y YOLOv8 |
| `inference.py` | Inferencia individual sobre una imagen (útil para depurar) |

## pipeline/

Pipeline modular reutilizable para entrenar nuevos modelos sobre nuevos datasets (por ejemplo, fase de tuberculosis sobre TBX11K o fase multipatología sobre VinDr-CXR 22 clases). Se organiza así:

```
pipeline/
├── configs/      YAML con hiperparámetros por experimento
├── data/         loaders y splits StratifiedGroupKFold
├── models/       builders de Faster R-CNN, YOLOv8 y RetinaNet
├── training/     bucles de entrenamiento, optimizadores y schedulers
├── evaluation/   FROC, métricas NODE21, exportación a TIDE
├── inference/    inferencia batch para conjuntos grandes
└── utils/        IO, transformaciones y logging
```

Para arrancar un experimento nuevo con esta pipeline, copia un `config.yaml` existente, ajusta `dataset:`, `model:` y `output_dir:`, y ejecuta:

```bash
cd ~/nodule_detection
python -m pipeline.training.train --config configs/<tu_experimento>.yaml
```

## Reproducir los resultados publicados

| Resultado | Script o pipeline | Tiempo estimado (GB10) |
|---|---|---|
| Ensemble WBF (NODE21 = 0.9391) | `scripts/ensemble_wbf.py` sobre dos checkpoints | minutos |
| Faster R-CNN VinDr corregido (NODE21 = 0.9025) | `scripts/train_frcnn_vindr.py` | ~3 h |
| YOLOv8s individual (NODE21 = 0.9103) | `scripts/train_yolo.py --model yolov8s --imgsz 1024` | ~2 h |
| Replicación Behrendt et al. | `scripts/train_frcnn_reproduce.py` | ~3 h |
| Cuantificación del data leakage | `scripts/generate_augmented_images.py` + comparativa con `evaluate.py` | minutos |

Los checkpoints binarios (`.pth`, `.pt`) están excluidos por `.gitignore`. Se distribuyen separadamente según se acuerde con los autores.
