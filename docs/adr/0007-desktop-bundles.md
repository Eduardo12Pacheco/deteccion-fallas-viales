# 0007 - Bundles de escritorio: lanzador PyInstaller onedir con modelo embebido

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisión D12 y S15 del plan; enmienda L5)
- **Tarea:** T13

## Contexto

S15 pide builds de escritorio portátiles en GitHub Releases que arranquen la misma
aplicación local y abran el navegador, sin una segunda cadena de herramientas (Tauri) y
sin que el usuario instale Python. PyInstaller onedir se eligió en D12 porque reutiliza
la pila de Python y evita las dependencias de webview de Linux. El bundle debe
funcionar sin conexión, embeber el modelo ONNX pasado a `build_desktop.py`, mantener
fuera el stack de entrenamiento (torch, scipy, matplotlib) y quedar por debajo del
presupuesto de 250 MB cuando se construye con el modelo pequeño de CI.

Dos restricciones del entorno moldearon el lanzador: L5 exige que todos los puertos
locales por defecto estén en 16000-16999 (así que el histórico 8765 pasó a 16080), y
el servidor uvicorn debe ejecutarse en un hilo worker mientras el lanzador maneja
Ctrl+C. PyInstaller 6.22 desactiva UPX en Linux, y la wheel actual de
`opencv-python-headless` 5.x aporta unos 165 MB sin comprimir (79 MB de `cv2.abi3.so`
más sus bibliotecas de códecs), lo que por sí solo superaría el presupuesto de tamaño
junto con Python, ONNX Runtime y numpy.

## Decisión

- **Punto de entrada y estructura.** `packaging/rdv_desktop.py` llama a
  `rdv.entrypoints.desktop.main`. `packaging/rdv.spec` construye un ejecutable onedir
  `RoadDamageVision` con `console=True`, los assets estáticos de la web, `models.json`,
  las muestras, los esquemas JSON y exactamente un modelo ONNX bajo
  `rdv/assets/models/`. `scripts/build_desktop.py` pasa el modelo mediante
  `RDV_DESKTOP_MODEL`, ejecuta PyInstaller en un directorio de trabajo temporal y
  archiva el resultado como `RoadDamageVision-<version>-<plataforma>.tar.gz` (`.zip` en
  Windows). `scripts/smoke_bundle.py` lanza el directorio construido, verifica
  `/api/health`, sube una muestra extraída del bundle, sondea el análisis hasta `done`,
  luego envía Ctrl+C y exige código de salida 0.
- **Lanzador.** `rdv.entrypoints.desktop` enlaza el primer puerto libre a partir de
  `16080` (hasta 50 intentos), arranca uvicorn en un hilo daemon, sondea `/api/health`
  durante hasta 60 s, muestra la URL y abre el navegador salvo que se pase
  `--no-browser`. También acepta `--host`, `--port` y `--model`. El modelo por defecto
  es el `assets/models/*.onnx` embebido; los ajustes `RDV_*` siguen aplicando, y una
  ruta de modelo incluida evita cualquier descarga.
- **Apagado ordenado.** El hilo del servidor se envuelve en un `ServerRunner` cuya
  finalización se señala con un `threading.Event`. Es deliberado: en CPython, una
  `KeyboardInterrupt` lanzada dentro de `threading.Thread.join` puede dejar
  `is_alive()`/`join()` en un estado engañoso (el hilo se reporta muerto mientras sigue
  ejecutándose), lo que antes permitía que el proceso terminara antes de que uvicorn
  ejecutara el shutdown del lifespan de FastAPI y borrara su directorio temporal de
  análisis. Ahora el lanzador establece `server.should_exit`, espera el evento, y el
  test de humo verifica que no sobreviva ningún directorio `rdv-web-*`.
- **Sin OpenCV en el bundle.** El bundle excluye `cv2` para cumplir la barrera rígida de
  250 MB; incluirlo produjo un bundle de 298 MB con binarios sin símbolos de depuración. El análisis de imágenes no
  se ve afectado (Pillow decodifica, y `preprocess.resize_linear` replica
  `cv2.INTER_LINEAR` en numpy, probado en unitarios contra OpenCV con una diferencia de
  un nivel de 8 bits). Las cargas de video en el bundle devuelven un 422 que explica la
  limitación en lugar de decir que el archivo está dañado. La instalación de la CLI L1
  sigue incluyendo OpenCV y conserva el soporte completo de video (S6/S12). Es un
  compromiso de empaquetado documentado, no un cambio en la aplicación web ni en la CLI.
- **Exclusiones y tamaño.** Las exclusiones de D12 (`torch`, `torchvision`,
  `transformers`, `scipy`, `matplotlib`, `tkinter`) se aplican en la especificación, además
  de las herramientas de desarrollo. El test de humo falla si se recolecta algún directorio
  prohibido o si la suma de tamaños de archivo llega a 250 MiB; el bundle con el modelo
  pequeño ronda los 165 MiB con esa medida.
- **Plataformas y confianza.** T15 construye Linux (`ubuntu-24.04`), Windows x64 y macOS
  arm64. Los binarios no están firmados: Gatekeeper de macOS y SmartScreen de Windows
  exigen que el usuario permita la aplicación, y el build de Linux necesita glibc 2.39+
  porque se construye en Ubuntu 24.04. Estos límites se declaran en la documentación de
  release en lugar de dejarse implícitos.

## Consecuencias

- La aplicación de escritorio es un producto de primera clase para el análisis de
  imágenes, con el modelo real embebido; el video sigue siendo una función de la CLI
  soportada por la instalación base.
- Futuras versiones de OpenCV podrían ser lo bastante pequeñas como para reactivar el
  video en el bundle; la verificación de tamaño en `smoke_bundle.py` convierte eso en
  una decisión medida.
- `build_desktop.py` necesita el archivo de modelo fijado y el grupo de dependencias
  `build` (`uv sync --group build`); nunca necesita los extras train/cuda.
- Una carrera de puertos entre `find_free_port` y el enlace de uvicorn es posible en
  principio; el lanzador entonces falla la espera de salud con un mensaje claro en lugar
  de quedarse colgado.
- El bundle solo escribe en el directorio temporal del sistema (trabajos de análisis y
  los logs de uvicorn impresos en stdout); no modifica su directorio de instalación y
  funciona sin conexión porque no intenta descargar ningún modelo.
