# 0011 - Semántica del análisis de video, límites del tracker y salidas atómicas

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisiones D20, D21 y especificación S6 del plan)
- **Tarea:** T7

## Contexto

La especificación S6 exige que `rdv detect <video>` escriba un MP4 anotado con el mismo
número de fotogramas que el origen, un documento de detecciones con resultados por
fotograma analizado más un resumen, y que falle de forma limpia ante una entrada truncada
o corrupta. Detectar cada fotograma es lento en CPU, y el tracker no puede saber cómo se
movió la cámara, así que el pipeline necesita una semántica explícita y documentada en
lugar de un comportamiento de mejor esfuerzo.

## Decisión

- **Paso de análisis.** Solo se ejecuta inferencia sobre cada `max(1, round(source_fps /
  analysis_fps))`-ésimo fotograma de origen (`--analysis-fps`, por defecto 5). El MP4
  anotado sigue conteniendo todos los fotogramas de origen; los fotogramas no analizados
  repiten las detecciones del fotograma analizado más reciente. Así la línea de tiempo de
  la salida queda alineada con el video de entrada y la inferencia solo se paga para los
  fotogramas muestreados.
- **Tracker.** `IouTracker` se ejecuta solo sobre los fotogramas analizados, por clase,
  con emparejamiento voraz por IoU (`iou_threshold=0.3`), `max_age=5` fotogramas analizados
  y `min_hits=2`. Una "instancia única" es un track que alcanzó `min_hits`; su id es estable
  y se registra en el documento de detecciones (`track_id`).
- **Sin compensación de movimiento de la cámara.** El tracker compara cajas en coordenadas
  de imagen. Las cámaras en movimiento, los paneos y el zoom cambian la posición de las
  cajas entre fotogramas analizados, así que los conteos de instancias únicas son
  aproximados y así se etiquetan en la documentación y en los informes. Nunca se presentan
  como conteos de tráfico ni como mediciones de severidad.
- **Cobertura.** La razón de cobertura del resumen es la media aritmética de la cobertura
  exacta por unión de cajas en cada fotograma analizado (`D21`), una estadística descriptiva
  en `[0, 1]`, nunca una puntuación de severidad.
- **`--max-frames N`.** Limita el número de fotogramas de origen leídos. El documento de
  detecciones registra `params.max_frames`, el indicador `truncated` del resumen, y el resumen
  en Markdown indica que la ejecución se detuvo antes de tiempo, de modo que un análisis
  parcial nunca se confunda con uno completo.
- **Salidas atómicas.** El MP4 se codifica en `<stem>.annotated.part.mp4` y solo se renombra
  tras un flujo completo y exitoso; el JSON de detecciones y los resúmenes se escriben
  después. Una entrada truncada o corrupta lanza `MediaError` (código 5) y no deja ningún
  artefacto con su nombre final.
- **Guarda de número de fotogramas.** Cuando el contenedor declara un número de fotogramas,
  un flujo que termina antes es un error en lugar de un análisis silenciosamente corto.
  Los contenedores sin número de fotogramas se analizan hasta el fin del flujo.
- **Informes.** `rdv detect` escribe `<stem>.annotated.mp4`, `<stem>.detections.json`,
  `<stem>.summary.json` y `<stem>.summary.md` por video, de modo que varios videos en una
  misma invocación no se sobrescriban entre sí.

## Consecuencias

- El costo de inferencia en CPU escala con `analysis_fps`, no con la tasa de fotogramas del
  origen, mientras que los videos de salida siguen siendo exactos fotograma a fotograma.
- Los clips muy cortos pueden reportar cero instancias únicas cuando menos de dos fotogramas
  analizados contienen el objeto (`min_hits=2`); las detecciones crudas se siguen reportando.
- Las detecciones en fotogramas omitidos son una ayuda visual, no predicciones
  independientes; el JSON solo lista los fotogramas analizados para que las métricas y las
  auditorías no sean ambiguas.
- `rdv benchmark` mide el rendimiento de video como fotogramas analizados por segundo sobre
  fotogramas sintéticos de 320x240, independientemente de la decodificación del contenedor.
