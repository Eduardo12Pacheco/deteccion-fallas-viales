# 0002 - Subconjunto del dataset, postura de licencia y splits propios

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisiones D2, D3, D4, D15 del plan)
- **Tarea:** T5

## Contexto

RDD2022 (publicado a través de CRDDC'2022, artículo de Figshare 21431547) es un
conjunto de datos multinacional de daños en carretera con cuatro clases (`D00`
grieta longitudinal, `D10` grieta transversal, `D20` grieta de piel de cocodrilo,
`D40` bache). Es grande (47,420 imágenes en siete carpetas por país) y heterogéneo:
el sensor, la resolución y el punto de vista de captura cambian según el país, y
las anotaciones de test nunca se publicaron.

El registro de Figshare declara **CC BY 4.0**. La página del paper en arXiv y
algunos mirrors indican **CC BY-SA 4.0** (la licencia del propio paper), y las
imágenes de Estados Unidos provienen de Google Street View, cuyos términos son
separados de la licencia del dataset. Redistribuir un pequeño subconjunto de
imágenes requiere, por tanto, una política explícita y conservadora.

## Decisión

- **Entrenar solo con Japón, India y Chequia** (21,041 imágenes de train etiquetadas:
  10,506 / 7,706 / 2,829). Las tres son capturas a nivel de calle con teléfono
  inteligente o cámara de tablero con puntos de vista compatibles. China_Drone y
  China_MotorBike usan otros sensores y quedan como trabajo futuro.
- **Nunca entrenar con United_States**, porque las imágenes de Google Street View
  tienen derechos de terceros que la licencia de Figshare no puede resolver. La
  carpeta sigue fijada en `data/rdd2022.lock.json` como procedencia, pero no se
  descarga por defecto.
- **Usar Noruega solo como muestra de evaluación fuera de dominio** (1,000 imágenes
  de train con semilla, nunca usadas para entrenar). Noruega mide unos 3650x2044 y
  es una buena sonda del cambio de dominio.
- **Licenciar los fixtures derivados como CC BY-SA 4.0.** El código del repositorio
  sigue siendo MIT; `tests/fixtures/rdd2022-mini` y `src/rdv/assets/samples` llevan
  un aviso CC BY-SA 4.0 y un `ATTRIBUTION.md` que nombra las imágenes de origen, el
  DOI y el paper. Esto satisface ambas lecturas de la licencia del dataset. La
  cuestión legal abierta sobre modelos entrenados con datos CC se declara en la
  model card (T10/T16).
- **Crear splits propios.** Las anotaciones públicas de test no existen, así que
  usar las carpetas de test oficiales dejaría nada que medir. En su lugar, se
  ordenan los ids de train por país, se cortan bloques de 100 ids consecutivos, se
  mezclan los bloques con `random.Random(42)` y se asignan de forma voraz a los
  objetivos 80/10/10. Los bloques mantienen en la misma partición las capturas
  consecutivas casi duplicadas tanto como sea posible; la fuga residual es una
  limitación documentada (D3).
- **Conservar las anotaciones vacías como imágenes negativas** (cerca de 70 % de los
  XML de Chequia no tienen objetos), porque los falsos positivos sobre carretera
  intacta importan en un detector.

## Consecuencias

- El conjunto de entrenamiento es lo bastante homogéneo para ajustar un solo modelo
  y, al mismo tiempo, es multipaís. Cada métrica reportada se mide sobre datos que el
  modelo nunca vio.
- El mini fixture y las muestras son redistribuibles legalmente con atribución, pero
  son un subconjunto muy pequeño: para entrenar, los usuarios deben descargar el
  dataset completo.
- El número de Noruega es una señal fuera de dominio, no parte de las métricas
  objetivo dentro de dominio.
- Los splits son reproducibles a partir del árbol descargado y están fijados por
  sha256, de modo que un split cambiado se ve en la revisión y no altera los
  resultados en silencio.
