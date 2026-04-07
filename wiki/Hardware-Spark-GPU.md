# Hardware Spark GPU

[← Home](Home.md)

Todos los experimentos y el worker de inferencia del prototipo corren sobre una **NVIDIA DGX Spark**, una plataforma compacta destinada a desarrollo de IA basada en el superchip Grace Hopper GB10. Es una plataforma poco habitual y poco documentada en literatura de deep learning para imagen médica, por lo que la sección dedicada en la memoria registra los detalles relevantes para futuros lectores.

## Especificaciones

| Componente | Valor |
|---|---|
| CPU | NVIDIA Grace · 20 cores ARM Neoverse V2 |
| GPU | NVIDIA Blackwell (Grace Hopper GB10) |
| Arquitectura | aarch64 (ARM64) — **no x86** |
| Memoria unificada | 128 GB LPDDR5X, accesible por CPU y GPU |
| Bandwidth memoria | 273 GB/s |
| Almacenamiento | 4 TB NVMe |
| Sistema operativo | Ubuntu 24.04 LTS aarch64 |
| CUDA toolkit | 13.0 |
| Driver NVIDIA | Open-source (KO) compatible Grace |
| PyTorch | 2.x (build aarch64) |
| Python | 3.11 (venv aislado) |

La arquitectura aarch64 implica que **no es válido instalar wheels precompiladas para x86**. PyTorch, torchvision, Ultralytics y SimpleITK requieren builds nativos o ruedas precompiladas para ARM. Algunas dependencias menos comunes (por ejemplo `pycocotools`) se compilan desde fuente.

## Acceso

El servidor está en la red interna del centro. Acceso vía SSH:

```bash
ssh ${SPARK_USER}@${SPARK_HOST}
```

Las credenciales reales no están en este repositorio. Variables de entorno habituales:

| Variable | Significado |
|---|---|
| `SPARK_USER` | Nombre de usuario Unix en la Spark |
| `SPARK_HOST` | Hostname interno (resolución por DNS local) |
| `SPARK_KEY` | Ruta a la clave SSH (fuera del repo) |

## Estructura de directorios

Convenio adoptado en la Spark para mantener consistencia:

```
~/nodule_detection/
├── pipeline/        clone de spark/pipeline/ de este repo
├── scripts/         clone de spark/scripts/
├── worker/          clone de spark/worker/
├── checkpoints/     pesos entrenados (no versionados)
├── weights/         pesos preentrenados (VinDr backbone)
├── data/            dataset NODE21 (PNGs preprocesados)
├── logs/            logs de entrenamiento
└── venv/            virtualenv aislado de Python
```

## Comandos básicos

### Estado de la GPU

```bash
nvidia-smi
```

Salida esperada: GPU Grace Hopper visible, memoria utilizada, procesos en curso.

### Activar el entorno

```bash
cd ~/nodule_detection
source venv/bin/activate
export PYTHONPATH=$(pwd)
```

### Arrancar el worker manualmente

```bash
./worker/start_worker.sh
```

### Servicio systemd

El worker está empaquetado como unidad systemd ([`spark/worker/cxr-worker.service`](../spark/worker/cxr-worker.service)). Para instalar:

```bash
sudo cp ~/nodule_detection/worker/cxr-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable cxr-worker
sudo systemctl start cxr-worker
```

Comandos útiles:

```bash
sudo systemctl status cxr-worker        # estado
sudo systemctl restart cxr-worker       # reiniciar
sudo journalctl -u cxr-worker -f        # logs en vivo
```

### Logs del worker

```bash
tail -f ~/nodule_detection/worker/worker_live.log
```

### Lanzar un entrenamiento

```bash
cd ~/nodule_detection
nohup python scripts/train_frcnn_vindr.py --config configs/frcnn_vindr.yaml \
  > logs/frcnn_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## Particularidades aarch64

- **Wheels precompiladas:** PyTorch publica wheels aarch64 en su canal nightly; usar `pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130`. Ultralytics y SimpleITK proporcionan wheels arm64 oficiales.
- **TensorRT:** disponible con build nativo aarch64, no usado en este proyecto (preferimos PyTorch para flexibilidad).
- **Docker:** Docker funciona, pero la mayoría de imágenes oficiales son x86. Para CXR Detection no es necesario Docker en la Spark (el worker corre nativo).
- **Memoria unificada:** la coherencia CPU-GPU reduce la latencia de transferencia, pero hay que tener cuidado con la fragmentación. Liberar tensores con `torch.cuda.empty_cache()` al final de cada batch grande.

## Rendimiento observado

| Tarea | Tiempo |
|---|---|
| Inferencia Faster R-CNN sobre 1 imagen 1024×1024 | 55 ms |
| Inferencia YOLOv8s sobre 1 imagen 1024×1024 | 25 ms |
| Inferencia ensemble + WBF | 81 ms |
| Entrenamiento Faster R-CNN 30 épocas sobre NODE21 | ~3 h |
| Entrenamiento YOLOv8s 100 épocas sobre NODE21 | ~2 h |
| Preprocesado MHA → PNG (4 882 imágenes) | ~15 min |

## Backup

Política operativa: snapshot completo de `~/nodule_detection/` antes de cada experimento significativo:

```bash
rsync -av --delete ~/nodule_detection/ ~/nodule_detection_backup_$(date +%Y%m%d)/
```

El backup permite revertir hiperparámetros o checkpoints sin recurrir a git en la Spark (que sólo tracea código, no pesos ni datos).
