# Evaluación

`rdv evaluate` ejecuta un detector sobre un split etiquetado, calcula métricas de detección
de objetos de estilo COCO con el motor de dominio puro y escribe un directorio de informe.
Nunca usa el split de test para ajustar nada.

## Comando

```bash
rdv evaluate \
  --data data/rdd2022 \
  --splits data/rdd2022/splits.json \
  --split test \
  --model rdv-rtdetrv2-r18-rdd2022 \
  --out reports/v1
```

Opciones:

| Opción | Significado |
|--------|-------------|
| `--data DIR` | Raíz del dataset que contiene `<Country>/train/{images,annotations/xmls}`. |
| `--splits FILE` | Manifiesto de splits; por defecto `<data>/splits.json`. |
| `--split {val,test,all}` | Split etiquetado a evaluar (el contrato de la CLI en S7). |
| `--model ID\|PATH` | Id de modelo del manifiesto o un ONNX local de un solo archivo. |
| `--out DIR` | Directorio de salida (por defecto `reports/v1`). |
| `--op-threshold auto\|F` | `auto` selecciona el óptimo de F1 en `val`; un número se usa tal cual. |
| `--ood DIR` | Directorio VOC opcional fuera de dominio (muestra de Noruega). |
| `--max-images N` | Evalúa solo los primeros N ids después de ordenar. |
| `--device auto\|cpu\|cuda` | Dispositivo de ejecución. |

Las detecciones se recogen con un corte de puntaje de `0.001` para que el motor de métricas
vea la distribución completa ordenada por puntaje. El **umbral de operación** es independiente:
se usa solo para la matriz de confusión, la taxonomía de errores y la galería.

## Política de umbral

El umbral de operación es el puntaje que maximiza el F1 global con IoU 0.5, calculado
agrupando todas las clases y todas las imágenes del split de **validación** (decisión D10).
El split de test nunca se lee durante la selección; un test espía del repositorio verifica que
el repositorio de validación se consume por completo antes de abrir el split evaluado. Con
`--op-threshold F` el split de validación no se abre en absoluto. Los informes registran
`operating_source` como `val` o `explicit`.

## Semántica de las métricas

- mAP@0.5 y mAP@0.5:0.95 usan los 10 umbrales de IoU de COCO y la envolvente de precisión
  interpolada de 101 puntos, con un máximo de 100 detecciones por imagen y clase.
- El AP por clase se informa para D00, D10, D20 y D40. Una clase sin cajas de ground truth en
  el split evaluado aporta `null` y se excluye de la media, igual que en `pycocotools`.
- Las métricas por país agrupan los ids de imagen por su primer token (el nombre del país en
  los ids de RDD2022), de modo que reutilizan el mismo motor sin cambios.
- Cuando se pasa `--ood`, las mismas métricas se calculan sobre ese conjunto independiente y se
  guardan bajo `ood`.
- Las métricas se redondean a seis decimales; el JSON se escribe con las claves ordenadas, de
  modo que dos corridas idénticas producen bytes idénticos.

## Taxonomía de errores (D10)

El emparejamiento es independiente de la clase y voraz por puntaje descendente con IoU 0.5. Cada
predicción no emparejada con un puntaje igual o superior al umbral de operación se clasifica como:

| Categoría | Regla (se evalúa en este orden) |
|-----------|---------------------------------|
| `classification` | IoU >= 0.5 con una caja de ground truth de otra clase. |
| `localization` | `0.1 <= IoU < 0.5` con una caja de ground truth de la misma clase. |
| `duplicate` | IoU >= 0.5 con una caja de la misma clase ya emparejada por un puntaje mayor. |
| `background` | Cualquier otro caso. |

Las cajas de ground truth no emparejadas se cuentan como `missed`. `error_analysis.json` guarda
los conteos y cada caso con su id de imagen, clase, puntaje, caja y la caja de ground truth
emparejada cuando corresponde.

## Artefactos

| Archivo | Contenido |
|---------|-----------|
| `metrics.json` | `rdv.metrics/v1`: identidad del modelo, conteos del dataset, mAP/por clase, punto de operación, matriz de confusión, conteos de la taxonomía, y métricas por país y OOD. |
| `report.md` | Informe legible con todas las tablas y enlaces a los demás artefactos. |
| `report.html` | Informe autocontenido: CSS en línea, gráficos y miniaturas de la galería en base64, autoescape de Jinja2. |
| `pr_curves.png` | Curvas precisión-recall con IoU 0.5 para las cuatro clases más la media. |
| `pr_curve_<CLS>.csv` | Traza cruda `recall,precision,score` por clase. |
| `confusion_matrix.png` / `.csv` | Matriz 5x5 con fondo, en el umbral de operación. |
| `error_analysis.json` | Taxonomía `rdv.error-analysis/v1` más cada caso. |
| `gallery/<category>/*.jpg` | Peores casos seleccionados con las cajas de ground truth y de predicción. |
| `gallery/index.html` | Índice de la galería; solo enlaza las imágenes escritas en la misma corrida. |

## Selección de la galería

Los falsos positivos se ordenan por puntaje y los omitidos por área del ground truth. Se conservan
como máximo seis casos por categoría y la lista final se limita a 24. Una pasada por turnos
garantiza que cada categoría no vacía esté representada antes de que alguna reciba su sexto caso,
de modo que la galería nunca oculte un tipo de error completo detrás del límite. La selección es
determinista: los empates se resuelven por id de imagen, id de clase y coordenadas de la caja.

## Determinismo y limitaciones

- No se escriben marcas de tiempo ni rutas absolutas en `metrics.json`.
- El motor de métricas se verifica contra `pycocotools` en la suite de tests `parity`; ver
  `docs/adr/0005-own-metrics-engine.md`.
- Las anotaciones de test de RDD2022 no son públicas, así que el proyecto evalúa sobre un split
  por bloques con semilla de las carpetas de train etiquetadas (decisión D3). Puede haber fuga
  residual de casi duplicados entre bloques, y se indica en la documentación del dataset.
- Noruega es solo fuera de dominio: nunca se usa para entrenar, y sus imágenes provienen de otras
  cámaras y otras resoluciones.
- Estas métricas son números de investigación, no una certificación de seguridad crítica.
