# 0003 - Arquitectura hexagonal y su verificación

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisiones D5, D9 y S16 del plan)
- **Tarea:** T4

## Contexto

El proyecto combina un dominio de píxeles puro (cajas, decodificación, splits,
métricas), varios adaptadores de E/S (datasets VOC, inferencia ONNX/PyTorch, media,
reportes) y tres entrypoints (CLI con Typer, aplicación web local y lanzador de
escritorio). Sin una regla estricta, el dominio importaría Pillow, ONNX Runtime o
FastAPI "solo esta vez". Eso impediría probar la lógica costosa sin esas pilas y
bloquearía la instalación base sin torch que exige la especificación S2.

El plan exige arquitectura hexagonal: `entrypoints -> composition -> adapters ->
application -> domain`, con el dominio limitado a la biblioteca estándar más NumPy
y con los adaptadores prohibidos de importarse entre sí (D9).

## Decisión

Adoptar puertos y adaptadores con una raíz de composición explícita y hacer cumplir
la dirección de dependencias de forma mecánica:

- **Dominio** (`rdv.domain`): objetos de valor inmutables y algoritmos puros, solo
  biblioteca estándar y NumPy. Sin E/S, sin logging, sin settings.
- **Aplicación** (`rdv.application`): casos de uso y puertos `typing.Protocol` en
  `ports.py`. Solo importa el dominio. Los puertos describen capacidades, no
  implementaciones: los primeros son `DatasetRepository` (iteración de metadatos
  más `load_image` perezoso) y `ProgressSink` (observador de los valores
  `ProgressEvent` de `events.py`). `NullProgressSink` es el Null Object que se usa
  cuando no hay observador conectado.
- **Adaptadores** (`rdv.adapters.*`): implementaciones dirigidas (driven) de los
  puertos. Pueden importar bibliotecas de terceros y el dominio/la aplicación, pero
  no entre sí; solo `rdv.adapters.training` puede importar torch, transformers o
  scipy. `Rdd2022VocRepository` y el parser VOC viven en `rdv.adapters.datasets`.
- **Composición** (`rdv.composition`): el único módulo que conoce los adaptadores
  concretos; los construye a partir de `rdv.config.Settings` tipados.
- **Entrypoints** (`rdv.entrypoints.*`): CLI, web y escritorio. Traducen la entrada
  del usuario en llamadas a casos de uso y muestran los resultados. Nunca
  instancian un adaptador dirigido directamente.

Los contratos se verifican con `import-linter` (`layers`, `forbidden`,
`independence` en `pyproject.toml`) y con un test AST de pureza que recorre
`src/rdv/domain` y rechaza cualquier import fuera de la biblioteca estándar y NumPy.

## Consecuencias

- Las pruebas del dominio se ejecutan en milisegundos y no pueden tocar la red, el
  sistema de archivos ni una GPU por accidente. El motor de métricas puede
  compararse con `pycocotools` en un entorno aislado.
- La instalación base sigue sin torch (S2), porque la única ruta de import de torch
  es `adapters.training`, que carga de forma perezosa la raíz de composición.
- Agregar una fuente de datos o un backend de modelo significa agregar un adaptador
  que cumpla un protocolo; el código de aplicación no cambia. El repositorio y el
  observador de progreso añadidos aquí son las primeras dos de esas costuras.
- La raíz de composición es el único lugar que puede crecer a medida que aparecen
  adaptadores en tareas posteriores; mantenerla solo con adaptadores lo cubre el
  contrato de capas.
- Las clases Protocol son estructurales: los adaptadores no heredan de ellas, así
  que un error tipográfico lo detecta `mypy --strict` en el punto de llamada y no
  al importar. Los tests verifican el comportamiento a través de la API pública con
  forma de protocolo.
