# Model card — `rdv-rtdetrv2-r18-rdd2022` v1

- **Estado:** artefacto publicado, primer ajuste fino real (T10)
- **Artefacto:** `rdv-rtdetrv2-r18-rdd2022-v1.onnx`, ONNX de un solo archivo, opset 17, sha256
  `806be021998a850eb074892b0025e7b6361b771fe773378191039af513b9a7de`, 80,760,832 bytes
- **Referencia de métricas:** [`reports/v1/metrics.json`](../reports/v1/metrics.json)
- **Licencia del código:** MIT (este repositorio). **Pesos base:** Apache-2.0. **Derivados del dataset:**
  postura CC BY-SA 4.0 (ver [DATA.md](DATA.md)); el estado de licencia de los propios pesos ajustados se
  discute en [Notas éticas y legales](#notas-éticas-y-legales).

## Detalles del modelo

| Campo | Valor |
|---|---|
| Arquitectura | RT-DETRv2 R18vd (`RTDetrV2ForObjectDetection`, Hugging Face Transformers 4.57.6) |
| Checkpoint base | `PekingU/rtdetr_v2_r18vd`, revisión `5650961749fa93567c0d46fc7f43ea4f9e914107`, `model.safetensors` sha256 `d18309d0d7ea57048138885c4c6ecfcb1e24506fc6153b94ad484f8ab62c7115` |
| Licencia base | Apache-2.0 (model card y metadatos de la API de Hugging Face) |
| Entrada | RGB, estirado a 640x640, `pixel_values [N,3,640,640]` float32, rango 0-1, sin normalización media/desviación |
| Salidas | `logits [N,300,4]` (antes de sigmoide) y `pred_boxes [N,300,4]` (cxcywh normalizado al lienzo de entrada estirado) |
| Clases | `D00` grieta longitudinal, `D10` grieta transversal, `D20` grieta de piel de cocodrilo, `D40` bache (ids 0..3) |
| Parámetros | unos 20 M (backbone R18 + codificador/decodificador RT-DETR) |
| Runtime | ONNX Runtime CPU por defecto; inferencia opcional PyTorch/CUDA dentro del entorno de entrenamiento |

El grafo ONNX se exportó desde el checkpoint EMA ajustado con `torch.onnx.export` (ruta TorchScript
heredada, opset 17) y se volvió a guardar como un solo archivo. Una comprobación de paridad entre torch y
ONNX Runtime en ocho imágenes de validación midió una diferencia máxima de filas emparejadas de `3.2e-5`
para logits y cajas, por debajo del contrato de `1e-3`. En una de esas ocho imágenes, el orden de empate
del TopK de puntajes exactos difiere entre la ejecución eager y la trazada; las salidas son un conjunto de
300 consultas, y el emparejamiento biyectivo de filas confirma que los valores coinciden.

## Uso previsto

- Demostración local y sin conexión de la detección de daños en carretera en imágenes fijas y video,
  incluidas la interfaz web `rdv serve` y el bundle de escritorio portátil.
- Uso de investigación/portafolio, evaluación reproducible y punto de partida para un detector de un
  dominio específico.
- Exploración interactiva: un deslizador de confianza y los filtros por clase permiten a un usuario
  intercambiar precisión por exhaustividad sin volver a ejecutar la inferencia.

## Fuera de alcance

- Decisiones de seguridad crítica, legales, de seguros o de prioridad de mantenimiento. El modelo no está
  validado para esos usos y no debe guiarlos.
- Índices de estado del pavimento (PCI/IRI), puntuación de severidad, segmentación, reconstrucción de
  profundidad o 3D, mapeo GPS o localización a nivel de carril.
- Despliegue en producción o en vehículos, garantías de tiempo real, o afirmaciones de precisión de estado
  del arte.
- Imágenes de los subconjuntos excluidos de RDD2022 (Google Street View de Estados Unidos, China
  Drone/MotorBike) y fotografía que no sea de daños en carretera.

## Datos de entrenamiento

RDD2022 (CRDDC'2022), artículo de Figshare `21431547`, DOI `10.6084/m9.figshare.21431547.v1`. Solo se
descargaron las carpetas `train/` etiquetadas de Japón, India y Chequia (unos 1.9 GB mediante lecturas
por rangos de ZIP anidados) y se dividieron con la regla de bloques con semilla de D3:

| Split | Imágenes | Japón | India | Chequia |
|---|---:|---:|---:|---:|
| train | 16,741 | 8,406 | 6,106 | 2,229 |
| val | 2,200 | 1,100 | 800 | 300 |
| test | 2,100 | 1,000 | 800 | 300 |
| OOD (Noruega, nunca entrenado) | 1,000 | — | — | — |

- Semilla 42, tamaño de bloque 100, proporciones 0.8/0.1/0.1 por país (total 21,041).
- Las imágenes sin objetos se conservan como negativos; las etiquetas fuera de `{D00,D10,D20,D40}` se
  descartan y se cuentan (`docs/DATA.md`).
- Las anotaciones oficiales de test de RDD2022 no son públicas; el split de test anterior es el split
  retenido propio de este proyecto, no el test del desafío CRDDC.
- La muestra de Noruega es solo un conjunto fuera de dominio: imágenes de 3,650x2,044 estiradas a 640x640,
  nunca usadas para entrenamiento ni para seleccionar el umbral.
- Los datos de Estados Unidos están excluidos (términos de Google Street View); China Drone/MotorBike están
  excluidos (sensores y puntos de vista distintos).

El mini fixture versionado (`tests/fixtures/rdd2022-mini`, 16 imágenes) y las seis muestras empaquetadas se
redistribuyen bajo la postura CC BY-SA 4.0 documentada en [DATA.md](DATA.md).

## Procedimiento de entrenamiento

| Ajuste | Valor |
|---|---|
| Épocas | 24 (regla de presupuesto D16; `26.01 img/s` medidos en `--benchmark-only`) |
| Batch efectivo | 24 = micro-batch 12 x acumulación de gradiente 2 (`ceil(16/12)`) |
| Optimizador | AdamW, lr de cabeza `1e-4`, lr de backbone `1e-5`, decaimiento de peso `1e-4` (sin decaimiento en normalización/sesgos), recorte de gradiente `0.1` |
| Programación | calentamiento lineal de 500 pasos + decaimiento coseno hasta 1 % |
| Precisión | autocast bf16 en CUDA |
| EMA | `0.9999`, calentamiento `min(0.9999, (1+t)/(10+t))` |
| Augmentation | resize estirado a 640x640, hflip 0.5, distorsión fotométrica 0.5, afín p=0.5 (traslación 0.1, escala 0.8-1.2), `SanitizeBoundingBoxes(min_size=2)` |
| Hardware | NVIDIA GeForce RTX 4060 8 GB, driver 620.70.06, runtime CUDA 13.0, CPU de 12 hilos |
| VRAM asignada pico | 5.82 GB (tope de la sonda de micro-batch automático 7.2 GB; el batch 16 no cabía) |
| Tiempo de reloj | 14,660 s (4.07 h) para 24 épocas, incluida la validación |
| Semillas | 42 (Python, NumPy, torch, workers del DataLoader) |

El registro de la corrida (git sha, hash de configuración, versiones, tiempo de reloj, memoria pico) está
versionado como [`reports/v1/run.json`](../reports/v1/run.json); el registro por época es
[`reports/v1/epochs.csv`](../reports/v1/epochs.csv). El artefacto final es la EMA de la última época, que
obtuvo en validación un resultado ligeramente mejor que el checkpoint del mejor peso crudo (mAP@0.5:0.95
0.1690 frente a 0.1661); ambos se midieron en el split de validación y se publicó la EMA mejor.

## Evaluación

Split de test retenido dentro del dominio (Japón + India + Chequia, 2,100 imágenes, 2,469 cajas de ground
truth). Métricas de estilo COCO del motor numpy del proyecto, 101 puntos de recall, maxDets 100. La matriz
de confusión y la taxonomía de errores usan el umbral de operación seleccionado en el split de validación,
nunca en test.

| Métrica | Valor |
|---|---:|
| mAP@0.5 | **0.3794** |
| mAP@0.5:0.95 | **0.1696** |
| AP@0.5 D00 grieta longitudinal | 0.3383 |
| AP@0.5 D10 grieta transversal | 0.3089 |
| AP@0.5 D20 grieta de piel de cocodrilo | 0.4955 |
| AP@0.5 D40 bache | 0.3749 |
| Umbral de operación óptimo para F1 (val) | 0.0216 |
| Precisión / recall en ese umbral | 0.425 / 0.448 |

Barra mínima S10 (mAP@0.5 >= 0.35, mAP@0.5:0.95 >= 0.15, cada clase AP@0.5 >= 0.10): **cumplida**;
`scripts/render_readme_metrics.py --status` muestra `PASS (>= min bar)`.

### Calibración de la confianza

La cabeza v1 tiene confianzas bajas: la distribución de puntajes llega como máximo a unos 0.06 en las
muestras empaquetadas, y el umbral óptimo para F1 en validación es 0.0216. Un valor por defecto de "0.25 por
convención" no mostraría ninguna detección en las muestras de demostración. El `recommended_score_threshold`
del manifiesto es **0.01**, el punto más bajo en el que las seis muestras empaquetadas siguen produciendo al
menos una detección; los usuarios pueden subir el deslizador de la interfaz para obtener menos cajas, pero
con más confianza. Ninguna muestra empaquetada alcanzó el margen de 0.15 de T10 por encima del umbral, y
`reports/v1/sample_expectations.json` registra ese hecho de forma explícita en lugar de descartar todas las
muestras. Es una limitación de calibración, no una edición de métricas: los valores de mAP informados se
basan en el ranking y no cambian.

### Resultados por país en test

| País | Imágenes | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|
| Japón | 1,000 | 0.4261 | 0.1994 |
| India | 800 | 0.2666 | 0.1189 |
| Chequia | 300 | 0.2780 | 0.1014 |

El AP@0.5 de `D10` en India cae a `0.0002` (Japón aporta el 0.3089 global), y el subconjunto de Chequia es
pequeño; el número global oculta esta brecha de dominio. El desglose completo por clase está en
[`reports/v1/per_country.json`](../reports/v1/per_country.json).

### Fuera de dominio (Noruega)

| Métrica | Valor |
|---|---:|
| Imágenes / cajas | 1,000 / 1,193 |
| mAP@0.5 | 0.0588 |
| mAP@0.5:0.95 | 0.0229 |

La caída grande es la esperada: Noruega usa otra cámara, otra resolución y otra apariencia de carretera, y
nunca se entrenó con ella. El modelo no se presenta como geográficamente general.

### Perfil de errores en el umbral de operación

| Categoría | Conteo |
|---|---:|
| Omitidos (FN) | 1,441 |
| Localización (FP) | 378 |
| Fondo (FP) | 760 |
| Duplicado (FP) | 121 |
| Clasificación (FP) | 62 |

Los veinticuatro peores casos (seis por categoría, con tope de 24) están versionados en
[`reports/v1/gallery/`](../reports/v1/gallery/), con la taxonomía en
[`reports/v1/error_analysis.json`](../reports/v1/error_analysis.json).

## Rendimiento de inferencia

Medido por `rdv benchmark` sobre el artefacto (batch 1, entrada 640x640, 20 pasadas de calentamiento + 100
pasadas cronometradas): CPU p50 **156.3 ms** (p95 210.1 ms, Intel i5-12400T de 12.ª generación, ONNX
Runtime) y GPU p50 **14.1 ms** (p95 16.1 ms, RTX 4060, torch bf16). Rendimiento de video sintético de 320x240
en CPU: **6.09 fotogramas analizados/s**. Ambas compuertas de cordura S11 pasan (CPU <= 1500 ms, GPU <= 100 ms).
Hardware completo y configuración de la corrida: [`reports/v1/benchmark.json`](../reports/v1/benchmark.json).

## Limitaciones

- **No es de seguridad crítica.** La precisión está lejos de una herramienta de inspección en producción; los
  falsos negativos son el error dominante.
- **Cambio de dominio.** El mAP@0.5 de Noruega es 0.0588; el `D10` de India está cerca de cero en el split de
  test. No extrapoles a países, cámaras o resoluciones no vistas.
- **Split propio, no el test oficial del desafío.** Las anotaciones oficiales de test de RDD2022 nunca se
  publicaron.
- **Advertencia sobre el split por bloques.** Los bloques de ids consecutivos reducen, pero no eliminan, la
  fuga de fotogramas casi duplicados entre splits.
- **Confusión entre tipos de grieta.** Las grietas longitudinales y transversales pueden parecerse a 640x640,
  y el modelo no se entrenó con las etiquetas de RDD2022 descartadas (D43/D44/D50/D01/D11/D0w0).
- **Los conteos en video son aproximados.** Las instancias únicas provienen de un tracker IoU voraz sin
  compensación del movimiento de la cámara (D20/ADR 0011).
- **La razón de cobertura es descriptiva**, no una puntuación de severidad ni PCI (D21).
- **Compromiso de modelo pequeño.** RT-DETRv2 R18 se eligió por una GPU de 8 GB y por licencias permisivas, no
  por máxima precisión.
- **Confianzas bajas.** Los puntajes absolutos son bajos (máximos de muestra <= 0.062), así que el umbral
  recomendado es 0.01 y no el 0.25 convencional. Usa el ranking o la confianza relativa, o recalibra antes de
  cualquier regla posterior que suponga probabilidades calibradas.

## Notas éticas y legales

- Las imágenes de carretera pueden contener rostros, placas de matrícula e indicios de ubicación. Las muestras
  empaquetadas y las imágenes del fixture son fotos reales de carretera; se incluyen solo con fines de
  demostración (postura CC BY-SA 4.0, atribución en [DATA.md](DATA.md)).
- Las predicciones no deben usarse para negar reclamaciones, priorizar reparaciones ni tomar decisiones
  presupuestarias sin inspección humana.
- El registro del dataset dice CC BY 4.0 mientras el paper y algunos mirrors dicen CC BY-SA 4.0; este proyecto
  sigue la lectura conservadora (atribuir a los autores y redistribuir las imágenes derivadas bajo CC BY-SA
  4.0) y documenta la discrepancia en lugar de resolverla en silencio.
- La situación legal de los pesos generativos o de modelo derivados de imágenes con licencia CC no está
  resuelta. Los pesos base son Apache-2.0; los pesos ajustados se distribuyen con fines de portafolio y
  demostración con esta advertencia, sin afirmar una titularidad limpia.

## Cita

Si usas los datos o la arquitectura base, cita:

- Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Sekimoto, Y. (2022).
  RDD2022: A multi-national image dataset for automatic road damage detection.
  arXiv:2209.08538.
- Lyu, W. et al. (2024). RT-DETRv2: Improved Real-time Detection Transformer.
  arXiv:2407.17140.
