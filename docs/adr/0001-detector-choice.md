# 0001 - Elección del detector: RT-DETRv2 R18vd y evitar AGPL

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisión D1 del plan; especificaciones S3, S4, S9 y S10)
- **Tarea:** T6 (inferencia y exportación), T16 (registro)

## Contexto

El proyecto necesita un detector de daños en carretera preentrenado que se pueda
ajustar (fine-tune) en una GPU de consumo de 8 GB, exportar a un único archivo
ONNX y redistribuir como parte de un repositorio de portafolio con licencia MIT.
Los pesos base deben tener una licencia permisiva explícita, porque el repositorio
publica los pesos ajustados como asset de release y los distribuye dentro de los
bundles de escritorio.

La opción evidente, Ultralytics YOLO (YOLOv5/YOLOv8/YOLO11 y su envoltorio de
RT-DETR), tiene licencia **AGPL-3.0** según los metadatos de GitHub y de PyPI. Un
servicio en red o un derivado redistribuido bajo AGPL-3.0 obligaría a todo el
proyecto a adoptar esa licencia, lo que contradice el requisito MIT de S3. Por eso
esa opción se descartó antes de escribir la ruta de inferencia.

Además, el detector debe cumplir el contrato de despliegue de S4: un único grafo
ONNX de 640x640 con `logits [N,300,4]` y `pred_boxes [N,300,4]`, sin NMS, y una
ruta de exportación mantenida por una biblioteca muy usada.

## Decisión

Usar **RT-DETRv2 R18vd mediante Hugging Face Transformers**:

- checkpoint `PekingU/rtdetr_v2_r18vd`, revisión
  `5650961749fa93567c0d46fc7f43ea4f9e914107`;
- `model.safetensors` de 80,904,640 bytes, sha256
  `d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115`, fijado en
  `configs/train/rdd2022-rtdetrv2-r18.yaml` y verificado antes del entrenamiento;
- Apache-2.0 para código y pesos (`cardData.license` de la API de Hugging Face y
  front matter de la model card), preentrenado en COCO;
- unos 20 M de parámetros, de extremo a extremo (sin NMS), entrada fija de
  640x640, exportado en T9/T10 con la escalera de respaldo D8 (opset 18, opset 17,
  TorchScript heredado);
- ONNX Runtime CPU como motor de inferencia por defecto; PyTorch/CUDA bf16 solo
  dentro del entorno de entrenamiento (ADR 0004).

Si la exportación v2 hubiera fallado tras los intentos de D8, el respaldo
configurado era la familia v1 `PekingU/rtdetr_r18vd` (también Apache-2.0),
seleccionable solo mediante `model.hf_id`. La exportación tuvo éxito, así que no
hizo falta cambiar.

### Alternativas descartadas

| Opción | Evidencia de licencia | Motivo del descarte |
|---|---|---|
| Ultralytics YOLOv5/v8/YOLO11 y su envoltorio RT-DETR | AGPL-3.0 (API de GitHub y metadatos de PyPI) | Copyleft incompatible con el código MIT y con los pesos redistribuidos |
| Modelos de detección de `torchvision` | Código BSD-3-Clause; sin declaración de licencia para los pesos preentrenados (el README pide determinar el permiso) | La ambigüedad de la licencia de los pesos hace insegura la redistribución |
| YOLOX | Apache-2.0, pero la última release en PyPI es la 0.3.0 de 2022-04 | Empaquetado desactualizado y herramientas de exportación sin mantenimiento |
| RF-DETR | Repositorio Apache-2.0, árbol de dependencias más pesado, licencias por checkpoint no verificadas | Dependencias adicionales y términos de los pesos sin verificar |
| Modelos de segmentación o de índices de pavimento | — | Fuera de alcance: el proyecto detecta cajas, no máscaras ni severidad |

## Consecuencias

- El ajuste fino se mantiene dentro del presupuesto de 8 GB de VRAM: pico medido de
  5.8182 GB en la RTX 4060 con micro-batch 12, 24 épocas en 4.07 h (ver
  [MODEL_CARD.md](../MODEL_CARD.md)).
- La instalación base no incluye torch; la inferencia depende solo de ONNX Runtime
  y numpy, y el bundle de escritorio se mantiene por debajo de la barrera de 250 MiB
  (ADR 0007).
- La cabeza de clasificación COCO se reinicializa para las cuatro clases de RDD2022
  (`ignore_mismatched_sizes=True`). El grafo exportado tiene una salida fija de 300
  consultas, que consume el decodificador en numpy puro de `rdv.domain.decoding`.
- La precisión es honesta pero modesta (mAP@0.5 de test 0.3794, mAP@0.5:0.95
  0.1696), y la cabeza v1 tiene confianzas bajas (umbral recomendado 0.01). La model
  card y el README publican esos números sin cambios.
- El cumplimiento de la licencia se verifica con `tests/unit/test_license_guard.py`
  (ninguna dependencia de runtime AGPL/GPL) y con un escaneo del código fuente que
  solo permite el nombre de Ultralytics en los ADR y en `docs/DATA.md`.
- El nombre Ultralytics permanece en este registro de decisión y en la tabla de
  procedencia de datos, tal como permite S3: la opción descartada se documenta en
  lugar de ocultarse.

## Referencias

- Model card y API de Hugging Face: <https://huggingface.co/PekingU/rtdetr_v2_r18vd>
- Artículo de RT-DETRv2: Lyu, W. et al. (2024), arXiv:2407.17140
- Licencia de Ultralytics: <https://github.com/ultralytics/ultralytics>
- Nota de licencia de los modelos preentrenados de torchvision: <https://github.com/pytorch/vision>
- Decisiones de datos y entrenamiento: [DATA.md](../DATA.md), [TRAINING.md](../TRAINING.md)
