# 0004 - Instalación base sin torch y ONNX Runtime como motor por defecto

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisiones D7, D8, D9 del plan; S2, S4)
- **Tarea:** T6, T9

## Contexto

El repositorio debe instalarse y realizar inferencia en Linux, Windows y macOS con
un único `uv tool install` (especificación S1), mientras que el stack de
entrenamiento (PyTorch, torchvision, Transformers, SciPy) es grande, depende de la
plataforma y solo se necesita para reproducir el ajuste fino. Incluir esos paquetes
en la instalación base inflaría el entorno a varios GB, impediría mantener el bundle
de escritorio bajo el límite de tamaño y arrastraría wheels de CUDA a máquinas que
solo necesitan inferencia en CPU. La especificación S2 fija el requisito estricto:
después de importar la CLI, `torch`, `torchvision`, `transformers` y `scipy` no deben
cargarse; importar el adaptador de inferencia no debe cargar `matplotlib`.

El detector debe seguir siendo desplegable con un contrato de tensores bien definido
(especificación S4), para que el mismo archivo lo usen la CLI, la aplicación web, el
bundle de escritorio y el benchmark.

## Decisión

Mantener el conjunto de dependencias base sin torch y exponer el stack de
entrenamiento como extras opcionales:

- `train` agrega `transformers`, `scipy`, `pyyaml` y `huggingface_hub`.
- `cpu`/`cuda` agregan `torch` y `torchvision` desde los índices explícitos de
  PyTorch; los dos extras son mutuamente excluyentes (`[tool.uv] conflicts`).
- La instalación base contiene ONNX Runtime, NumPy, Pillow, OpenCV-headless, FastAPI
  y el stack de reportes.
- La inferencia usa por defecto `OnnxDetector`: un grafo ONNX de un solo archivo
  cuya entrada es `pixel_values` `float32 [N,3,640,640]` en `[0,1]` y cuyas salidas
  son `logits [N,300,4]` y `pred_boxes [N,300,4]` (`cx,cy,w,h` normalizados). El
  preprocesamiento es una sola función (`rdv.adapters.inference.preprocess`) y la
  decodificación permanece en el dominio puro (`rdv.domain.decoding`).
- La selección de dispositivo es una Strategy (`rdv.adapters.inference.providers`):
  `RDV_DEVICE=auto` elige CUDA solo cuando ONNX Runtime expone
  `CUDAExecutionProvider`; en otro caso, CPU. Una solicitud explícita de `cuda` sin
  ese proveedor lanza `ModelError` (código de salida 4) en lugar de ejecutarse en
  silencio en CPU. `onnxruntime-gpu` se documenta como opcional porque entra en
  conflicto con la wheel base `onnxruntime`.
- El contrato de import-linter prohíbe `torch`, `torchvision`, `transformers` y
  `scipy` en todas partes excepto en `rdv.adapters.training`, de modo que el límite
  se verifica con un test y no por convención.
- La única excepción nombrada es la carga perezosa. `rdv.composition` (la sonda de
  CUDA y la fábrica opcional del detector de GPU) y
  `rdv.adapters.inference.torch_detector` solo pueden nombrar `torch`,
  `torchvision` o `transformers` a través de `importlib`. import-linter no ve las
  llamadas a importlib, así que
  `tests/unit/test_architecture.py::test_torch_stack_is_loaded_only_by_allowed_modules`
  recorre el AST de cada módulo y falla ante cualquier import estático de esos
  paquetes en otro módulo, o ante una llamada literal a `importlib.import_module` /
  `__import__`.
- `TorchDetector` (T9) vive en `rdv.adapters.inference` (el árbol de módulos del
  plan), pero importa el stack opcional mediante `importlib` al construirse, así que
  el contrato de imports estáticos sigue siendo válido. Sin el extra, lanza
  `MissingExtraError` (código de salida 6) con la indicación de instalación.
- Los pesos de entrenamiento se fijan por sha256 y se verifican antes de
  instanciar el modelo. La exportación siempre pasa por un envoltorio que fija el
  eje de consultas en 300 y vuelve a guardar un único archivo; la ruta TorchScript
  heredada es el último respaldo de opset para la versión fijada `transformers<5`.

## Consecuencias

- `uv tool install` produce una CLI utilizable en una máquina limpia sin GPU, sin
  CUDA toolkit ni descargas de varios GB.
- La CLI y el adaptador de inferencia pueden importarse en tests y en el bundle de
  escritorio sin arrastrar el stack de entrenamiento; la especificación S2 se verifica
  con un test de integración que importa la CLI y revisa `sys.modules`.
- El backend torch (`TorchDetector`, T9) es un adaptador alternativo detrás del mismo
  puerto `Detector`; el resto de la aplicación no cambia.
- La CLI sigue siendo importable en un entorno solo base, incluidos `rdv train --help`
  y `rdv export --help`; solo al ejecutarlos se carga el stack opcional.
- El soporte de GPU con ONNX Runtime es opcional y CI no lo ejercita. El benchmark
  reporta números de CPU por defecto y números de GPU solo cuando está instalado el
  extra CUDA de torch.
