# 0008 - Distribución del modelo mediante un manifiesto fijado y un almacén reanudable

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisiones D13, D22 y S20 del plan)
- **Tarea:** T6 (almacén y CLI), T10 (primer artefacto publicado)

## Contexto

La wheel debe mantenerse pequeña y el repositorio nunca debe versionar pesos de
modelo. Al mismo tiempo, `rdv detect` tiene que funcionar sin conexión una vez que hay
un modelo presente, no debe descargar nada de forma implícita (especificación S20) y
debe poder demostrar que el archivo que ejecuta es exactamente el artefacto con el que
se produjeron las métricas del README. Los pesos se producen después, en la GPU del
responsable, y se publican como asset de GitHub Release (`models-v1`).

## Decisión

- `src/rdv/assets/models.json` (esquema `rdv.models/v1`) es la fuente única de verdad
  para los artefactos descargables. Cada entrada fija `id`, `version`, `url`, `sha256`,
  `size`, `input_size`, `recommended_score_threshold`, `classes`, `license` y un
  `metrics_ref` opcional. La lista queda vacía hasta que T10 publique el primer ajuste
  fino.
- `rdv models list|pull|path|verify` es la única superficie que descarga o valida
  pesos:
  - `pull` transmite a `<name>.part`, se reanuda con una petición HTTP `Range` cuando
    existe un archivo parcial (reinicia desde cero si el servidor ignora el rango),
    verifica el tamaño exacto en bytes y el sha256, y solo después renombra el archivo
    de forma atómica. Un sha256 no coincidente elimina el archivo parcial y sale con
    código 4.
  - `RDV_OFFLINE=1` hace que `pull` falle antes de cualquier petición HTTP.
  - `resolve` acepta un id del manifiesto (buscado en `RDV_MODEL_DIR`) o una ruta
    `.onnx` explícita, de modo que los modelos locales y los publicados comparten la
    misma ruta de código.
  - `verify` compara un archivo con su entrada del manifiesto (o busca la entrada cuyo
    digest coincida) y devuelve esa entrada.
- `rdv detect` resuelve el modelo a través del mismo almacén. Un modelo ausente es un
  `ModelError` (código 4) cuyo mensaje apunta a `rdv models pull`; no hay descarga
  silenciosa.
- El umbral recomendado viene del manifiesto; `RDV_SCORE_THRESHOLD` lo sobrescribe y
  `--conf` sobrescribe ambos. Sin entrada en el manifiesto (archivo local), el respaldo
  documentado es `0.25`.

## Consecuencias

- El repositorio, la wheel y el bundle de escritorio no contienen pesos; el bundle
  embebe el artefacto en el momento de construirlo (T13) después de verificar este mismo
  sha256.
- Cada documento de detecciones registra el id del modelo y su sha256, de modo que un
  informe puede rastrearse hasta el artefacto exacto.
- Las descargas son idempotentes: una entrada de caché verificada evita la red, y un
  fallo de red conserva el `.part` para el siguiente intento.
- Para una publicación pública no se requiere ninguna clave. Como el repositorio del
  portafolio es privado, `pull` también acepta un `GITHUB_TOKEN`/`GH_TOKEN` opcional
  (por ejemplo, `gh auth token`). GitHub responde 404 a la URL de descarga del navegador
  de un release privado incluso con un token bearer; por eso, cuando hay token, el
  almacén resuelve la URL estable del manifiesto a través de la API de Releases y
  descarga el asset con `Accept: application/octet-stream`. El encabezado bearer solo se
  envía a hosts de GitHub. Las peticiones anónimas no cambian, y los releases públicos
  siguen funcionando sin token.
- El manifiesto es lo bastante pequeño como para revisarlo a mano, lo que mantiene visible
  el ancla de confianza en el repositorio en lugar de ocultarla en los metadatos del
  release.
