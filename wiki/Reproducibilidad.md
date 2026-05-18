# Reproducibilidad

[← Home](Home.md)

Cuatro escenarios prácticos para reutilizar el material de este repositorio. Para cada uno: requisitos, comandos, tiempo estimado y posibles problemas.

---

## A · Compilar la memoria académica

**Qué obtienes:** el PDF `memoria/Memoria.pdf` regenerado desde fuente.

**Requisitos:**

- Distribución LaTeX moderna: TeX Live ≥ 2023 (Linux/macOS) o MiKTeX ≥ 23 (Windows).
- Paquetes: `IEEEtran`, `amsmath`, `graphicx`, `hyperref`, `xcolor`, `listings`, `biblatex` (con motor `biber`).
- Espacio: ~ 100 MB para auxiliares LaTeX.

**Comandos:**

```bash
cd memoria/source
latexmk -pdf Memoria_v2_IEEE.tex
cp Memoria_v2_IEEE.pdf ../Memoria.pdf
```

**Tiempo estimado:** 2–3 minutos en hardware moderno.

**Problemas habituales:**

- *Falta `IEEEtran.cls`:* `tlmgr install ieeetran` o, en MiKTeX, activar instalación automática de paquetes.
- *Falta `biber`:* `tlmgr install biber` o instalar `biber` desde el sistema (`apt install biber` en Debian/Ubuntu).
- *Caracteres unicode en `.bib`:* compilar con `pdflatex -enable-write18` y motor `biber`, no `bibtex`.

---

## B · Reproducir el entrenamiento desde cero

**Qué obtienes:** los checkpoints `frcnn_corrected/best_node21.pth` y `yolo/yolov8s/best.pt` desde dataset crudo.

**Requisitos:**

- Acceso al dataset NODE21 (registro en <https://node21.grand-challenge.org/>).
- GPU NVIDIA con ≥ 16 GB VRAM (probado en Grace Hopper GB10, debería funcionar en RTX 3090, RTX 4090 o A100).
- Python 3.11 con dependencias: `torch`, `torchvision`, `ultralytics`, `albumentations`, `SimpleITK`, `pycocotools`, `ensemble-boxes`.
- Espacio: ~ 50 GB (dataset + augmentaciones + checkpoints intermedios).

**Comandos:**

```bash
# 1. Preprocesado MHA → PNG (15 min)
python spark/scripts/preprocess_node21.py \
  --input  /path/to/node21_raw/ \
  --output ~/nodule_detection/data/png_images/

# 2. Generación de augmentaciones offline (5 min)
python spark/scripts/generate_augmented_images.py \
  --input  ~/nodule_detection/data/png_images/ \
  --output ~/nodule_detection/data/augmented/

# 3. Entrenamiento Faster R-CNN con preentrenamiento VinDr (3 h)
python spark/scripts/train_frcnn_vindr.py \
  --vindr-weights ~/nodule_detection/weights/frcnn_vindr_pretrained.pth \
  --output-dir    ~/nodule_detection/checkpoints/frcnn_corrected/

# 4. Entrenamiento YOLOv8s (2 h)
python spark/scripts/train_yolo.py \
  --model yolov8s \
  --imgsz 1024 \
  --data  ~/nodule_detection/data/node21_yolo.yaml \
  --epochs 100

# 5. Evaluar ensemble (minutos)
python spark/scripts/ensemble_wbf.py \
  --frcnn-weights  ~/nodule_detection/checkpoints/frcnn_corrected/best_node21.pth \
  --yolo-weights   ~/nodule_detection/checkpoints/yolo/yolov8s/best.pt \
  --test-data      ~/nodule_detection/data/png_images_test/
```

**Tiempo total estimado:** ~ 5–6 horas en Grace Hopper GB10. Estimación para RTX 3090: ~ 8–10 horas.

**Problemas habituales:**

- *VRAM insuficiente:* reducir `batch_size` y aumentar `gradient_accumulation_steps`. Faster R-CNN con `batch=4` y `imgsz=1024` requiere ~ 14 GB; YOLOv8s con `batch=16` requiere ~ 10 GB.
- *Pesos VinDr no disponibles:* el preentrenamiento sobre VinDr-CXR es el factor más determinante. Sin acceso a PhysioNet/VinDr, entrenar desde ImageNet pierde ~ 5 puntos NODE21.
- *No reproducción exacta de las cifras:* el seeding (`torch.manual_seed`, `numpy.random.seed`, `random.seed`) y el `cudnn.deterministic = True` están en los scripts, pero pequeñas diferencias entre versiones de PyTorch o CUDA pueden causar variaciones de ± 0.5 puntos NODE21.

---

## C · Evaluar los checkpoints existentes

**Qué obtienes:** las cifras NODE21, AUROC y CM reportadas en la memoria, partiendo de pesos ya entrenados.

**Requisitos:**

- Acceso a los checkpoints (no versionados en este repo; distribuir aparte).
- GPU con ≥ 8 GB VRAM.
- Dataset NODE21 con split de test ya preparado.

**Comandos:**

```bash
# Evaluación individual de cada modelo
python spark/scripts/evaluate.py \
  --model frcnn \
  --weights ~/nodule_detection/checkpoints/frcnn_corrected/best_node21.pth \
  --test-data ~/nodule_detection/data/png_images_test/

python spark/scripts/evaluate.py \
  --model yolov8 \
  --weights ~/nodule_detection/checkpoints/yolo/yolov8s/best.pt \
  --test-data ~/nodule_detection/data/png_images_test/

# Ensemble
python spark/scripts/ensemble_wbf.py \
  --frcnn-weights ~/nodule_detection/checkpoints/frcnn_corrected/best_node21.pth \
  --yolo-weights  ~/nodule_detection/checkpoints/yolo/yolov8s/best.pt \
  --test-data     ~/nodule_detection/data/png_images_test/
```

**Tiempo estimado:** 10–15 minutos por modelo (incluye generación de curva FROC + matriz de confusión + reporte CSV).

**Salida esperada:** los CSVs en `reports/` se actualizan con las mismas cifras que las publicadas en la memoria (ver [[Experimentos]]).

---

## D · Ejecutar el prototipo asistencial

**Qué obtienes:** el prototipo web completo corriendo en local, accesible vía navegador.

**Requisitos:**

- Docker Desktop (Windows / macOS) o Docker Engine (Linux).
- Node.js 20+ con npm.
- GPU para el worker (o se puede deshabilitar el worker para probar UI/UX).
- Repositorios `cxr-detection` y `cxr-frontend` clonados.

**Comandos:**

```bash
# 1. Backend (FastAPI + MySQL + RabbitMQ + Nginx en Docker)
git clone https://github.com/tcontesti/cxr-detection
cd cxr-detection
cp .env.example .env       # ajustar credenciales
docker compose up -d
# Verificar: http://localhost:9020/api/health

# 2. Frontend (Vue 3 + Vite)
git clone https://github.com/tcontesti/cxr-frontend
cd cxr-frontend
npm install
npm run dev
# Abrir: http://localhost:5177

# 3. Worker GPU (en la Spark o máquina con GPU)
cd spark/worker
# Ajustar config.yaml con RABBITMQ_HOST y RABBITMQ_PASS
./start_worker.sh
```

**Tiempo total estimado:** primera vez 15–20 minutos (incluye descarga de imágenes Docker y `npm install`). Arranques posteriores ~ 1 minuto.

**Problemas habituales:**

- *Worker no se conecta a RabbitMQ:* verificar que el contenedor `cxr-rabbitmq` esté arriba con `docker ps`, comprobar puerto 5672 abierto en el firewall si worker y backend están en máquinas distintas.
- *Frontend muestra "API offline":* el backend tarda ~ 30 s en arrancar la primera vez (espera de MySQL y RabbitMQ). Revisar `docker compose logs cxr-backend`.
- *El worker no carga los modelos:* las rutas en `worker/config.yaml` son absolutas y específicas de la Spark (`/home/$SPARK_USER/nodule_detection/...`). Adaptar al entorno local.
- *No tengo GPU:* el frontend y el backend funcionan sin worker (el estudio quedará en estado `queued`). Útil para revisar UI/UX y probar el módulo de validación con resultados precargados.

Más detalle de arquitectura en [[Prototipo-CXR-Detection]] y en [`docs/DOCUMENTACION_TECNICA.md`](../docs/DOCUMENTACION_TECNICA.md).
