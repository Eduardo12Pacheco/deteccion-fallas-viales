# 0009 - Acceso por rangos al archivo anidado de Figshare

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisiones D14 y S8 del plan)
- **Tarea:** T5

## Contexto

La descarga de RDD2022 es un único ZIP de 13,264,172,619 bytes
(`https://ndownloader.figshare.com/files/38030910`, md5 `b62bd51d2ffcfaa76c60f234f0cc2bb3`)
que contiene un ZIP por país. Los tres países de entrenamiento necesitan unos 1.9 GB;
el archivo de Noruega por sí solo pesa 10.6 GB y solo se usa para una muestra fuera de
dominio de 1,000 imágenes. Descargar 13 GB para leer 1.9 GB es un desperdicio, y
descargar 10.6 GB para leer 1,000 imágenes es inaceptable en un portátil.

El endpoint de descarga responde con un 302 a una URL firmada de S3/R2 que expira a los
diez segundos (`X-Amz-Expires=10`). Por lo tanto, un descargador por rangos ingenuo falla
a mitad de transferencia si no vuelve a resolver la URL.

## Decisión

Implementar `RemoteZipArchive` en `rdv.adapters.datasets.remote_zip` como un lector de
rangos de bytes con posicionamiento sobre el ZIP externo:

- Una **caché de bloques de 4 MiB** (LRU, 64 bloques) respalda los saltos pequeños, como
  el parser del directorio central de `zipfile`; las lecturas de al menos un bloque
  evitan la caché y piden el rango exacto, de modo que transmitir un miembro grande no
  multiplica el tráfico por el tamaño del bloque.
- Los miembros externos están **almacenados sin comprimir**, así que un archivo interno es
  un rango contiguo del archivo externo. `open_nested()` analiza el directorio central
  externo, lee el encabezado local para calcular el primer byte de datos y devuelve una
  ventana posicionable que `zipfile.ZipFile` puede leer. El directorio central de Noruega
  (1.67 MB) y los rangos seleccionados de imágenes/XML son los únicos bytes que se
  descargan.
- El endpoint estable se **vuelve a resolver con un TTL de 5 s** y cada vez que un GET por
  rango devuelve 403. Directrices: reintentar 403/429/5xx hasta cinco veces con retroceso
  exponencial; cualquier otro estado es un `DatasetError` (código 3).
- `RDV_OFFLINE=1` lanza una excepción en la construcción, antes de la primera petición (S20).
- Las descargas escriben `<Country>.zip.part`, se reanudan desde el tamaño del part
  existente, verifican tamaño exacto, CRC32 y (cuando está fijado) sha256, y solo
  entonces renombran al destino final. Una verificación fallida elimina el part.
- `data/rdd2022.lock.json` fija el md5/tamaño del ZIP externo y, por país, el nombre del
  miembro externo, el desplazamiento del encabezado local, el tamaño y el CRC32. La primera
  descarga exitosa registra el sha256 del archivo interno en el mismo lock, lo que hace más
  fuertes las verificaciones posteriores sin cambiar el camino de adquisición.
- `rdv data download --full` es el respaldo: descarga el ZIP externo, verifica su md5 y
  extrae localmente los miembros internos necesarios.

## Consecuencias

- Una instalación normal descarga unos 1.9 GB en lugar de 13 GB, y la muestra de Noruega
  cuesta aproximadamente el tamaño de las JPEG seleccionadas más la sobrecarga de rangos.
- Los miembros Zip64 (Noruega y United_States superan los 4 GiB o están más allá del
  desplazamiento de 4 GiB) los maneja el parser estándar de `zipfile`; el lock guarda el
  desplazamiento del encabezado local, no un desplazamiento de datos calculado, de modo que
  una longitud distinta del campo extra no puede desplazar la descarga en silencio.
- Las pruebas usan `httpx.MockTransport` con un ZIP anidado sintético, así que ningún job de
  CI toca Figshare y las rutas de reanudación, CRC y expiración se ejercitan sin red.
- El adaptador vive en la capa de adaptadores dirigidos y lo construye solo
  `composition.py`; la CLI nunca importa `httpx` ni construye el archivo por sí misma.
