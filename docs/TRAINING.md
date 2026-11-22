# Entrenamiento

Este documento describe el pipeline reproducible de ajuste fino y la corrida medida publicada en
`reports/v1`. La instalación base no usa torch; el entrenamiento necesita el extra `train` más uno
de los extras de PyTorch, que son mutuamente excluyentes.

## Entorno

```bash
uv sync --extra train --extra cuda    # the real run (RTX 4060)
uv sync --extra train --extra cpu     # CPU tests, smoke runs and dry runs
```

`rdv train` y `rdv export` lanzan `MissingExtraError` (código de salida 6) con la misma sugerencia
cuando el stack no está instalado. Medido en la máquina del responsable: torch `2.14.1+cu130`,
torchvision `0.29.1+cu130`, transformers `4.57.6`, runtime de CUDA 13.0, driver de NVIDIA 620.70.06,
RTX 4060 8 GB.

## Datos

Descarga y divide RDD2022 primero (ver [DATA.md](DATA.md)):

```bash
export RDV_DATA_DIR=data          # keeps ~2 GB out of git and ~/.local
uv run rdv data download --countries Japan India Czech
uv run rdv data download --ood-norway 1000
uv run rdv data verify
uv run rdv data prepare
```

El entrenador lee `<RDV_DATA_DIR>/rdd2022/splits.json` y las carpetas de train de cada país; la
validación usa el split `val` de esa misma raíz. El directorio de datos nunca se deriva de una ruta
de configuración de entrenamiento. Tamaños finales de los splits: train 16,741 (Japón 8,406 / India
6,106 / Chequia 2,229), val 2,200, test 2,100, OOD de Noruega 1,000.

## Configuración

`configs/train/rdd2022-rtdetrv2-r18.yaml` es la configuración final D16. Está completamente tipada
(`rdv.adapters.training.config.TrainConfig`, Pydantic con `extra="forbid"`) y admite la expansión
`${ENV_VAR}` en cualquier valor de cadena.

| Campo | Significado |
|-------|-------------|
| `model.hf_id`, `model.revision`, `model.weights_sha256` | Checkpoint base fijado. El cargador descarga `config.json`, `preprocessor_config.json` y `model.safetensors` y verifica el digest de los pesos antes de instanciar el modelo. Se acepta un directorio local para corridas offline o de prueba. |
| `train.epochs`, `train.effective_batch` | Duración de la corrida y objetivo de batch efectivo (la acumulación de gradiente mantiene al menos esta cantidad de muestras por paso). |
| `train.lr`, `train.backbone_lr` | Tasas de aprendizaje cabeza y backbone (D16). |
| `train.warmup_steps`, `train.lr_min_factor` | Calentamiento lineal seguido de decaimiento coseno. |
| `train.precision` | Autocast `bf16` en CUDA; CPU siempre usa float32. |
| `train.ema` | Decaimiento `0.9999` con la rampa de calentamiento `min(decay, (1+t)/(10+t))`. |
| `train.eval_every` | Intervalo de épocas para el mAP@0.5:0.95 de validación y los checkpoints `best/`. |
| `train.max_images_per_epoch` | Sobrescritura de la regla de presupuesto (null en esta corrida). |
| `augment.*` | Pipeline torchvision v2: resize estirado, hflip, distorsión fotométrica, transformación afín, vflip/rot90/zoom-out opcionales y luego `SanitizeBoundingBoxes(min_size=2)`. |
| `budget.vram_cap_gb`, `budget.max_hours` | Límite del micro-batch automático (7.2 GB) y regla de parada por tiempo de reloj (10 h). |

## Comandos

```bash
# Validate config, dataset and model instantiation without training.
uv run rdv train --config configs/train/rdd2022-rtdetrv2-r18.yaml --dry-run

# Measure img/s, chosen micro-batch and peak VRAM (100 steps, D16 budget rule).
uv run rdv train --config configs/train/rdd2022-rtdetrv2-r18.yaml \
  --run-dir runs --benchmark-only

# Fine-tune, resuming automatically from runs/v1/last.pt when present.
uv run rdv train --config configs/train/rdd2022-rtdetrv2-r18.yaml \
  --run-dir runs/v1 --resume auto

# Export the published EMA checkpoint to the S4 single-file ONNX graph.
uv run rdv export --checkpoint runs/v1/selected \
  --out artifacts/rdv-rtdetrv2-r18-rdd2022-v1.onnx

# Evaluate the test split and the out-of-domain Norway sample.
uv run rdv evaluate --data data/rdd2022 --splits data/rdd2022/splits.json \
  --split test --model artifacts/rdv-rtdetrv2-r18-rdd2022-v1.onnx \
  --op-threshold auto --ood data/rdd2022-ood-norway --out reports/v1

# Real-model checks (never in CI).
RDV_REAL_MODEL=artifacts/rdv-rtdetrv2-r18-rdd2022-v1.onnx \
RDV_REAL_CHECKPOINT=runs/v1/selected RDV_DATA_DIR=data \
  uv run pytest -m realmodel -q
```

`make reproduce` ejecuta los pasos de Datos y el comando `rdv evaluate` de arriba en ese orden, con
`RDV_DATA_DIR=data`. No entrena ni exporta: el archivo ONNX en `artifacts/` debe existir ya (lo
escribe `rdv export`). Descarga los archivos de cada país (unos 2 GB) y sobrescribe `reports/v1`
versionado, así que revisa el resultado con `git diff` antes de confirmar nada.

## Corrida medida (v1)

| Campo | Valor |
|-------|-------|
| Benchmark | micro-batch 12, 26.0115 img/s, pico 5.6607 GB (100 pasos) |
| Micro-batch elegido | 12 (16 no cabía); acumulación 2; batch efectivo 24 |
| Épocas | 24 (24 x 643.6 s = 15,446 s <= 36,000 s, así que no hubo reducción de épocas ni tope `max_images_per_epoch`) |
| Tiempo de reloj | 14,660 s = 4.07 h, incluida la validación cada 2 épocas |
| Pico de VRAM asignada | 5.8182 GB (límite 7.2 GB) |
| Mejor métrica de validación con pesos crudos | mAP@0.5:0.95 = 0.1684 en la época 22 |
| Checkpoint publicado | EMA de la época 24, mAP@0.5 de val = 0.3788 / mAP@0.5:0.95 = 0.1690 |
| Resultado de test | mAP@0.5 = 0.3794, mAP@0.5:0.95 = 0.1696 |
| Artefacto | 80,760,832 B, sha256 `806be021998a850eb074892b0025e7b6361b771fe773378191039af513b9a7de` |

El registro de la corrida es `runs/v1/run.json` (versionado sin cambios como
[`reports/v1/run.json`](../reports/v1/run.json)); `runs/v1/epochs.csv` es
[`reports/v1/epochs.csv`](../reports/v1/epochs.csv). El `git_sha` del registro es el commit desde
el que empezó la corrida (`4b3866d`).

La métrica de validación se calcula con los pesos crudos durante la corrida, así que `best/` sigue
la mejor época cruda mientras guarda su contraparte EMA. Tras el entrenamiento, tanto el `best/` EMA
guardado como la EMA de la época 24 se puntuaron en el split de validación; la EMA de la época 24
fue ligeramente mejor (0.3788/0.1690 frente a 0.3735/0.1661 de mAP@0.5 / mAP@0.5:0.95) y se exportó
como `runs/v1/selected`. Esta comparación se registra aquí en lugar de elegir una en silencio; ambos
números provienen del motor de métricas de `rdv evaluate`.

## Semántica del entrenamiento

- **Modelo.** `RTDetrV2ForObjectDetection.from_pretrained` con cuatro etiquetas (`D00`, `D10`, `D20`,
  `D40`); la cabeza de clasificación COCO de 80 clases se reinicializa con
  `ignore_mismatched_sizes=True`, y todo lo demás se carga desde el checkpoint fijado.
- **La configuración es obligatoria.** Descargar solo `model.safetensors` sin `config.json` haría que
  `transformers` usara la configuración por defecto (más grande) y reinicializara en silencio partes
  del backbone; por eso el cargador exige el snapshot fijado completo.
- **Dataset.** `VocTorchDataset` envuelve el repositorio VOC de forma perezosa, conserva las imágenes
  sin cajas como negativos y convierte las cajas al contrato de Hugging Face: `class_labels` más
  `cxcywh` normalizado en el lienzo estirado de 640x640.
- **Optimizador.** AdamW con tasas de aprendizaje separadas para backbone y cabeza, y sin decaimiento
  de peso en los sesgos ni en los parámetros de normalización.
- **Acumulación.** El micro-batch sale de `[16, 12, 8, 6, 4]`; en CUDA la escalera se prueba con 20
  pasos reales limitados por `budget.vram_cap_gb`, usando un dataset lo bastante grande para esos 20
  pasos. La acumulación de gradiente usa `ceil(effective_batch / micro_batch)`, de modo que el batch
  efectivo es de al menos 16 (12 -> 24 aquí). En CPU el micro-batch se limita a 4 para las pruebas de
  humo.
- **Checkpoints.** `last.pt` se escribe de forma atómica en cada frontera de época y guarda el modelo,
  el optimizador, la EMA, el estado del generador del DataLoader y todos los RNG, de modo que una
  corrida reanudada reproduce el siguiente paso exactamente (verificado con tolerancia 1e-5). `best/`
  contiene los pesos EMA guardados con `save_pretrained` cada vez que mejora el mAP@0.5:0.95 de
  validación. Los pesos nunca se versionan en git.
- **Logs.** `metrics.jsonl` (un objeto por paso del optimizador), `epochs.csv` (una fila por época) y
  `run.json` (git sha, hash de configuración, semilla, versiones, GPU, tiempo de reloj y memoria pico;
  se reescribe en cada época). `benchmark.json` y `dry_run.json` siguen el mismo diseño de directorio
  de corrida.
- **Semillas.** Python, NumPy y torch se inicializan desde `seed` (42 por defecto); los workers del
  DataLoader reciben `seed + worker_id`.

## Ensamblado del informe

`rdv evaluate` escribe `metrics.json`, `report.md`, `report.html`, `pr_curves.png`,
`pr_curve_<CLS>.csv`, `confusion_matrix.png/.csv`, `error_analysis.json` y la galería. El
`reports/v1` publicado contiene además:

- `per_country.json` y `ood_norway.json`: las secciones `per_country` y `ood` de `metrics.json`,
  extraídas a documentos independientes;
- `run.json` y `epochs.csv`: copiados desde `runs/v1/`;
- `benchmark.json`: la salida de `rdv benchmark` para el ONNX publicado;
- `sample_expectations.json`: clases y puntajes máximos en las seis muestras empaquetadas, usado por
  la prueba de humo posterior al release.

Para mantener el conjunto de informes versionado por debajo del presupuesto de 6 MB, las JPEG de la
galería se recodificaron a un lado largo de 480 px con calidad 80, y las trazas CSV crudas de PR se
adelgazaron a como máximo 4,000 puntos igualmente espaciados (se conservan ambos extremos). `metrics.json`,
los gráficos y la taxonomía de errores no cambiaron; las curvas PNG siguen mostrando la interpolación de
101 puntos. Ningún número usado por el README o por la model card depende de las filas adelgazadas.

## Pruebas de humo en CPU

Las pruebas marcadas con `train` instancian un RT-DETRv2 reducido (profundidades R18 `[1,1,1,1]`, anchos
16/32/64/128, `d_model=32`, 20 consultas) y verifican, en CPU:

- pérdida finita y disminución de la pérdida en un sobreajuste de cuatro imágenes;
- reproducibilidad exacta al reanudar desde un checkpoint, comparando la pérdida del siguiente paso;
- los lotes con todos los negativos no hacen fallar el proceso;
- la augmentation mantiene las cajas dentro del lienzo y descarta las degeneradas;
- la forma cerrada de la EMA y la escalera de auto-batch ante OOM;
- la exportación ONNX pasa `onnx.checker`, expone `pixel_values [N,3,640,640]` y
  `logits`/`pred_boxes` `[N,300,4]`, y coincide con torch dentro de `1e-3`.

Ejecútalas con `uv run pytest -m train -q`.

## Contrato de exportación

`rdv export` carga el checkpoint, envuelve el modelo para que el eje de consultas sea siempre 300 (el
modelo de prueba pequeño rellena desde 20; el modelo real ya tiene 300) e intenta `torch.onnx.export`
con opset 18, luego opset 17, y finalmente el exportador TorchScript heredado. El modelo real eligió la
ruta heredada con opset 17. El grafo se vuelve a guardar como un solo archivo, se ejecuta
`onnx.checker`, y la paridad entre torch y ONNX Runtime debe mantenerse por debajo de `1e-3` en logits y
cajas antes de que la CLI informe éxito (medido con entrada aleatoria: logits `1.96e-5`, cajas
`1.44e-5`). La prueba `realmodel` repite la comparación sobre ocho imágenes de validación (máximo de
filas emparejadas `3.2e-5`).

## Benchmark e inferencia en GPU

`rdv benchmark --model <onnx> --checkpoint runs/v1/selected` añade la fila de GPU con torch bf16; los
números publicados están en [`reports/v1/benchmark.json`](../reports/v1/benchmark.json): CPU p50 156.3 ms
(p95 210.1 ms), GPU p50 14.1 ms (p95 16.1 ms), video en CPU 6.09 fotogramas analizados/s. Ambas
compuertas S11 pasan. La ruta local del `.onnx` se empareja con el manifiesto por sha256, así que el
benchmark registra el `model.id` publicado y el umbral recomendado. `TorchDetector` carga un directorio
de checkpoint, usa autocast bf16 en CUDA y float32 en CPU, y comparte la decodificación en numpy puro con
el detector ONNX. El adaptador importa torch mediante `importlib`, porque el contrato de arquitectura solo
permite un import estático de torch dentro de `rdv.adapters.training`; el archivo sigue en
`rdv.adapters.inference` tal como especifica la arquitectura T9.

## Regla de presupuesto

Después de que `--benchmark-only` informe `img_per_s` y la memoria pico:

```
epoch_s = n_train / img_per_s                     # 16741 / 26.0115 = 643.6 s
epochs  = 24            if 24 * epoch_s <= 36000  # 15446 s -> 24
        = max(12, floor(36000 / epoch_s)) otherwise
if 12 * epoch_s > 36000: max_images_per_epoch = max(8000, floor(3000 * img_per_s))
```

No se aplicó ninguno de los dos respaldos: el YAML versionado mantiene `epochs: 24` y
`max_images_per_epoch: null`. La guarda de tiempo de reloj de 10 h en `budget.max_hours` no detuvo
nada (la corrida tomó 4.07 h). Se permite un reintento acotado de 12 épocas extra con `lr x 0.3`
cuando no se alcanza la barra S10; la corrida v1 cumplió la barra, así que no se ejecutó ningún
reintento.

## Limitaciones

- Las métricas publicadas provienen del split con semilla propio de este proyecto, no del test oficial
  de RDD2022 (anotaciones no publicadas).
- El cambio de dominio es real: el mAP@0.5 de Noruega es 0.0588 y el `D10` de India está cerca de cero;
  ver `docs/MODEL_CARD.md`.
- La augmentation es una receta de partida, no un estudio de ablación; los valores de D16 se publican como
  puntos de partida.
- No se usa compensación del movimiento propio de la cámara en el video (ver ADR 0011).
- Los pesos base preentrenados son Apache-2.0; los derivados de RDD2022 siguen la postura CC BY-SA 4.0
  documentada en [DATA.md](DATA.md).
