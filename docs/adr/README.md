# Registros de decisiones de arquitectura

Cada decisión significativa se registra como un documento breve con **Estado**,
**Contexto**, **Decisión** y **Consecuencias**. Los registros son solo de adición:
las decisiones reemplazadas conservan su archivo y lo indican en la línea de estado.

| ADR | Decisión | Estado | Tarea | Plan |
|-----|----------|--------|-------|------|
| [0001](0001-detector-choice.md) | RT-DETRv2 R18vd (Apache-2.0) como detector; se descartan Ultralytics (AGPL-3.0) y los pesos con términos poco claros | Aceptada | T6, T16 | D1 |
| [0002](0002-dataset-subset-licensing.md) | Entrenar con RDD2022 Japón/India/Chequia; postura conservadora CC BY-SA 4.0 para imágenes derivadas; splits propios con semilla por bloques | Aceptada | T5 | D2, D3, D4 |
| [0003](0003-hexagonal-architecture.md) | Puertos y adaptadores hexagonales con `entrypoints > composition > adapters > application > domain`, verificados con import-linter y un test AST | Aceptada | T4 | D5, D9 |
| [0004](0004-torch-free-base-onnx-runtime.md) | Instalación base sin torch, extras `cpu`/`cuda` mutuamente excluyentes, ONNX Runtime CPU como motor por defecto | Aceptada | T6, T9 | D7, D8 |
| [0005](0005-own-metrics-engine.md) | Motor COCO mAP en numpy dentro del dominio, verificado contra `pycocotools` con tolerancia `1e-4` | Aceptada | T3 | D10 |
| [0006](0006-local-web-ui.md) | FastAPI + módulos ES sin build, trabajos por sondeo, seguridad solo de loopback y puerto por defecto 16080 | Aceptada | T11 | D11, D19 |
| [0007](0007-desktop-bundles.md) | Bundles PyInstaller onedir con el modelo embebido, OpenCV excluido y barrera de tamaño de 250 MiB | Aceptada | T13 | D12 |
| [0008](0008-model-distribution.md) | Manifiesto `models.json` fijado, almacén reanudable verificado por sha256 y asset `models-v1` en release | Aceptada | T6, T10 | D13, D22 |
| [0009](0009-ranged-nested-zip.md) | Acceso por rangos HTTP al ZIP anidado de Figshare con renovación de URL firmada y CRC32 fijados | Aceptada | T5 | D14 |
| [0010](0010-honest-metrics-policy.md) | Números del README y de la model card generados desde `reports/v1`, nunca editados a mano; lo que queda bajo el umbral sigue visible | Aceptada | T10, T16 | D16, D17 |
| [0011](0011-video-semantics.md) | Paso de análisis, límites del tracker IoU voraz, conteos únicos aproximados y salidas de video atómicas | Aceptada | T7 | D20, D21 |

## Convenciones

- Nombre de archivo: `NNNN-titulo-en-kebab-case.md`, con cuatro dígitos y en orden
  creciente.
- El estado es uno de `Propuesta`, `Aceptada`, `Reemplazada por NNNN` u `Obsoleta`.
- Cada registro nombra la(s) decisión(es) del plan que materializa y la tarea que la
  implementó.
- La fuente vinculante es el plan de implementación `new-projects/road-damage-vision.md`;
  un ADR explica **por qué**, y el código y la documentación describen **qué**.
