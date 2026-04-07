# Pipeline de Deteccion de Patologias en Radiografias de Torax

## Estructura
```
pipeline/
├── configs/              # Configuraciones YAML por dataset
│   ├── node21.yaml       # Nodulos pulmonares (NODE21 challenge)
│   ├── vindr_22classes.yaml  # VinDr-CXR 22 patologias
│   └── tuberculosis.yaml # TBX11K tuberculosis
├── models/               # Constructores de modelos
│   ├── frcnn_builder.py  # Faster R-CNN (VinDr/COCO/scratch + CBAM)
│   └── yolo_builder.py   # YOLOv8 / YOLO26
├── data/                 # Dataset y preprocesamiento
│   └── dataset.py        # DetectionDataset generico (PNG/DICOM/MHA)
├── utils/                # Utilidades
│   ├── metrics.py        # FROC, NODE21, AUROC, CM, mAP
│   └── ensemble.py       # Weighted Box Fusion (WBF)
└── README.md
```

## Como entrenar en un nuevo dataset

1. Crear config YAML (copiar de configs/node21.yaml y modificar)
2. Preparar CSV con columnas: img_name, file_path, label, x, y, width, height
3. Ejecutar entrenamiento:
   ```bash
   # Faster R-CNN
   python scripts/train_frcnn_reproduce.py --csv data/mi_csv.csv --version B --fold 0

   # YOLO
   python scripts/train_yolo.py --model yolov8s.pt --data data/yolo_dataset/mi_dataset.yaml
   ```

## Modelos soportados
- Faster R-CNN + ResNet50-FPN (con/sin VinDr weights, con/sin CBAM)
- YOLOv8 (s, m, l, x)
- YOLO26 (s, m, l, x) - NMS-free, ProgLoss, STAL

## Resultados NODE21 (sin data leakage)

| Modelo | NODE21 Score | AUROC | CM | ms/img |
|--------|-------------|-------|----|--------|
| FRCNN VinDr (Corrected-B) | 0.9025 | 0.9460 | 0.9146 | 35 |
| YOLOv8s | 0.9103 | 0.9686 | 0.9283 | 16 |
| YOLO26s | 0.7929 | 0.9557 | 0.8754 | 19 |

## Pesos disponibles
- `weights/fastercnn50.pth` -- Faster R-CNN preentrenado en VinDr-CXR
- `checkpoints/frcnn_corrected/best_node21.pth` -- FRCNN sin leakage NODE21=0.9025
- `checkpoints/yolo/yolov8s/best.pt` -- YOLOv8s NODE21=0.9103

## Maquina
- NVIDIA GB10 (Blackwell), CUDA 13.0, arm64, 128GB RAM unificada
- PyTorch 2.11.0+cu130, Ultralytics 8.4.33
