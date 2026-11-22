# 0006 - Interfaz web local: FastAPI, trabajos por sondeo y seguridad solo de loopback

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisiones D11, D19 y especificaciones S12 y S14 del plan)
- **Tarea:** T11

## Contexto

`rdv serve` debe exponer el detector mediante una aplicación de navegador terminada,
sin pedir al usuario que instale Node, una base de datos ni un servicio en la nube.
El mismo servidor viaja dentro de los bundles de escritorio (D12), así que debe
arrancar en pocos cientos de milisegundos, nunca cargar el stack de torch y no dejar
archivos atrás. La herramienta también corre en un portátil compartido donde varios
proyectos de portafolio levantan servidores locales en paralelo, por eso L5 restringe
los puertos por defecto al rango 16000-16999 (`16080` por defecto) y los tests a ese
rango o al puerto 0.

La inferencia está limitada por CPU/GPU y no debe ejecutarse en el hilo de la
petición. Un bucle de sondeo desde el navegador es más simple y robusto que SSE o
WebSockets para una aplicación local de un solo usuario, y el documento de
detecciones ya es JSON plano.

## Decisión

- **Stack.** FastAPI + uvicorn sirven JSON y una página estática con módulos ES. El
  contrato del servidor del plan se implementa en `entrypoints/web`: `app.py` (fábrica
  + lifespan), `api.py` (rutas), `jobs.py` (worker, cola, TTL) y `security.py`
  (middlewares ASGI). Sin build de Node, sin framework de frontend, sin base de datos.
- **Modelo de trabajos.** `POST /api/analyses` valida la carga de forma síncrona, luego
  crea un trabajo en una raíz `mkdtemp` por proceso y lo encola. Un hilo worker daemon
  por proceso consume la cola, de modo que las peticiones concurrentes no pueden
  sobrecargar la GPU/CPU. El progreso es un Observer sobre los eventos `ProgressSink`
  existentes y se guarda como `{stage, current, total, fraction, message}`; los
  contadores nunca retroceden. `GET /api/analyses/{id}` devuelve estado, progreso y
  (al terminar) el documento `rdv.detections/v1` y el resumen en línea, que el
  navegador renderiza sin otra petición.
- **Ciclo de vida.** El TTL viene de `RDV_ANALYSIS_TTL_MIN` (por defecto 30). Un
  barredor en segundo plano elimina los trabajos terminados, `DELETE` elimina uno de
  inmediato y el apagado del lifespan elimina toda la raíz temporal. Los trabajos
  encolados o en ejecución no se barren.
- **Cargas.** Los archivos se guardan con un nombre fijo `source` dentro del directorio
  del trabajo; el nombre enviado por el cliente se sanea y solo se usa como etiqueta de
  visualización. El contenido se detecta decodificándolo con los adaptadores reales:
  imágenes con Pillow y videos con OpenCV. Lo que no se decodifica como ninguno de los
  dos responde 422.
- **Seguridad (S14).** `LocalAccessMiddleware` rechaza con 421 cualquier `Host` que no
  sea `localhost`, `127.0.0.1` o `[::1]` (protección contra DNS rebinding) y rechaza
  con 403 cualquier petición que modifique estado cuyo `Origin` no coincida con el
  `Host`. `UploadSizeLimitMiddleware` aplica `RDV_MAX_UPLOAD_MB` a partir de
  `Content-Length` y, en peticiones chunked, mientras transmite el cuerpo a un archivo
  temporal en disco; las peticiones demasiado grandes reciben 413 antes de que el
  parser multipart vea los datos. No se agrega ningún middleware ni encabezado CORS.
  Los parámetros de ruta se validan contra un patrón hexadecimal de id de trabajo, de
  modo que las rutas de petición nunca llegan al sistema de archivos.
- **Descargas.** `report.json` es el documento de detecciones guardado con el nombre
  de la carga sustituido por la ruta privada del servidor. `report.html` es
  autocontenido (CSS en línea y una imagen anotada en base64 para las imágenes).
  `annotated.png` existe para los trabajos de imagen; la superposición del video se
  dibuja en el cliente a partir del JSON, de modo que el servidor nunca transcodifica.
- **CLI.** `rdv serve [--host] [--port] [--model] [--no-browser]` construye el detector
  antes de enlazar el puerto (un modelo ausente falla rápido con código de salida 4 y
  la sugerencia `rdv models pull`), luego ejecuta uvicorn y abre el navegador salvo que
  se pase `--no-browser`. El valor por defecto es `127.0.0.1:16080`.

## Consecuencias

- Un análisis lento mantiene cortas las peticiones HTTP; la interfaz sondea cada 500 ms.
- El worker único serializa el uso de CPU/GPU, lo cual es deliberado en un portátil: una
  cola de análisis espera en lugar de competir por memoria.
- Hacer spool de las cargas chunked cuesta una copia extra en disco, pero acota la
  memoria y garantiza un 413 incluso cuando falta `Content-Length`.
- Por defecto solo funciona el acceso desde loopback. Es posible enlazar otra dirección
  con `--host`, pero la verificación de Host de S14 solo acepta nombres de loopback, así
  que el acceso remoto no está soportado a propósito.
- La página estática enviada en T11 es un marcador de posición; T12 la reemplaza por el
  visor completo y los estados de UI probados con Playwright. El contrato de la API se
  congela aquí para que el frontend pueda construirse contra él.
