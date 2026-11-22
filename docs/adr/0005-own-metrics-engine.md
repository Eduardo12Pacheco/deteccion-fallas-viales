# 0005 - Motor de métricas propio con paridad frente a pycocotools

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisiones D10 y S7 del plan)
- **Tarea:** T3

## Contexto

El informe de evaluación es la evidencia central de este portafolio: mAP@0.5,
mAP@0.5:0.95, AP por clase, curvas de precisión/recall, una matriz de confusión y
una taxonomía de errores de estilo TIDE. La implementación obvia es llamar a
`pycocotools`, pero es una extensión en C solo de desarrollo que arrastra una
cadena de compilación a cada plataforma soportada y oculta la semántica de
evaluación detrás de un objeto `COCOeval` opaco. El informe además necesita
desgloses por país, un umbral de operación elegido con datos de validación y casos
de error que `pycocotools` no expone.

El plan exige que el motor de métricas viva en el dominio puro (solo biblioteca
estándar y NumPy, decisión D9) y que coincida con la semántica de COCO dentro de
`1e-4` (decisión D10, especificación S7).

## Decisión

Implementar el motor de métricas en `rdv.domain.metrics` con NumPy y verificarlo
contra `pycocotools` en un test de integración marcado como `parity`:

- `matching.py` reproduce el emparejamiento voraz de COCO por imagen y por clase:
  detecciones ordenadas por puntaje descendente (estable), como máximo
  `maxDets=100` por imagen y clase, y cada detección toma la última caja de ground
  truth aún no emparejada cuyo IoU sea al menos el mejor actual. Un emparejamiento
  exactamente en el umbral cuenta; con IoU 1.0 la referencia limita el umbral a
  `1 - 1e-10`.
- `average_precision.py` calcula el AP interpolado de 101 puntos con la envolvente
  monótona de precisión de COCO y la guarda de denominador `np.spacing(1)`. Las
  categorías sin ground truth se excluyen de la media, no se puntúan con 0.
- `curves.py` expone la traza cruda, la traza de 101 puntos y los arreglos de
  puntajes, y selecciona el umbral de operación por F1 máximo con IoU 0.5 (primer
  máximo; gana el puntaje más alto).
- `confusion.py` construye una matriz 5x5 independiente de la clase (filas = ground
  truth, columnas = predicción, fondo al final) con emparejamiento voraz de IoU 0.5
  por puntaje.
- `error_analysis.py` clasifica los falsos positivos como clasificación (IoU >= 0.5
  con una caja de ground truth de otra clase), localización (misma clase con
  `0.1 <= IoU < 0.5`), duplicado (misma clase ya emparejada, IoU >= 0.5) o fondo, en
  ese orden. La lista D10 pone la clasificación primero, así que gana cuando una caja
  podría ser a la vez un error de clasificación y uno de duplicado.
- `evaluator.py` es el caso de uso público: `evaluate(gt, predictions,
  iou_thresholds, max_dets=100)` devuelve métricas por clase, mAP@0.5 y mAP@0.5:0.95,
  además de una curva PR con IoU 0.5 siempre (incluso cuando los umbrales del
  llamador omiten 0.5). `by_prefix` agrupa ids de imagen como `Japan_000001` para las
  métricas por país.

`pycocotools` sigue siendo una dependencia solo de desarrollo, usada por el test de
paridad; ningún código de runtime lo importa.

## Consecuencias

- La capa de reportes consume objetos de valor tipados e inmutables y puede
  serializar cualquier campo sin analizar los internos de `COCOeval`.
- La evaluación es determinista y comprobable en cualquier sistema operativo con una
  instalación base; no se necesita compilación en C en runtime.
- El motor tiene unas 400 líneas más tests; la paridad con COCO solo está demostrada
  para el subconjunto soportado (bbox, sin flags de crowd/ignore, sin rangos de área,
  `maxDets` configurable pero con un solo valor). Ese subconjunto es exactamente el que
  necesita la especificación S7; extenderlo exige extender antes el test de paridad.
- Las diferencias numéricas con COCO están acotadas por el redondeo de IoU y la guarda
  `np.spacing(1)`; la suite de paridad verifica `|delta| <= 1e-4` para mAP, mAP@0.5 y
  AP@0.5 por clase.
- El mini fixture de 16 imágenes se crea en T5. El test de paridad de T3 construye un
  fixture sintético compacto con la misma forma (3 cajas por clase, 4 negativos) y
  verifica la paridad del AP y una matriz de confusión identidad; T8 evalúa el fixture
  real de extremo a extremo.
