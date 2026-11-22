# 0010 - Métricas honestas: generar los números, nunca editarlos a mano

- **Estado:** Aceptada
- **Fecha:** 2026-11-20
- **Decisores:** responsable del proyecto (decisiones D16, D17 y especificaciones S10, S20 del plan)
- **Tarea:** T10 (primera corrida real), T16 (bloques del README)

## Contexto

La afirmación central del proyecto es la reproducibilidad. Un número de mAP escrito a mano
en un README puede desviarse del informe versionado, puede redondearse hacia arriba o puede
ocultar una corrida que no alcanzó el mínimo documentado. El plan fija la barra S10
(mAP@0.5 >= 0.35, mAP@0.5:0.95 >= 0.15, AP@0.5 de cada clase >= 0.10) y la política de
reintento D16, así que la presentación debe derivarse mecánicamente de
`reports/v1/metrics.json`.

## Decisión

- `reports/v1/` es la fuente única de verdad: `metrics.json` proviene de `rdv evaluate`,
  `benchmark.json` de `rdv benchmark` y `run.json` del entrenador. Estos archivos se
  versionan; los pesos, checkpoints y el artefacto ONNX no.
- `scripts/render_readme_metrics.py` genera el Markdown entre `<!-- metrics:start -->` y
  `<!-- metrics:end -->` en `README.md` y `README.en.md` a partir de esos informes. También
  evalúa S10 y muestra `PASS (>= min bar)` o `BELOW TARGET` con los valores que fallan.
- `--check` termina con código distinto de cero cuando un bloque del README difiere de lo
  que renderizarían los informes, de modo que CI puede rechazar un número editado.
- Una corrida por debajo de la barra se publica sin cambios con la etiqueta `BELOW TARGET`.
  Se permite un reintento acotado de D16 (12 épocas extra con lr x 0.3); ambas corridas se
  conservan en `runs/`, se evalúa la mejor y ninguna se "corrige" en silencio. Si el AP@0.5
  de una clase sigue por debajo de 0.10, la tarea se eleva como punto de decisión en lugar de
  relajar el criterio.
- El `metrics.json` versionado es el artefacto referenciado por el manifiesto del modelo
  (`metrics_ref`) y por la model card; los mismos valores alimentan las expectativas de
  muestra que usa la prueba de humo posterior al release.
- `--status` es el comando que se usa en el Log de la tarea, de modo que el estado informado y
  el estado del README no pueden contradecirse.

## Consecuencias

- Los números del README pueden regenerarse desde el repositorio sin GPU.
- Un bloque de README desactualizado es un fallo de test, no una preferencia de documentación.
- Las métricas por debajo de la barra siguen visibles; el repositorio documenta lo que el
  modelo logró realmente en lugar de afirmar el objetivo.
- El ruido de coma flotante entre máquinas se gestiona en la capa de reproducción
  (`docs/TRAINING.md`, D23), no aflojando el renderizador.
