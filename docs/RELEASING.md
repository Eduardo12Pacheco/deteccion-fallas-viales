# Publicar road-damage-vision

Este proyecto publica dos canales de release independientes (decisión D22):

| Canal | Tag | Assets | Propósito |
|-------|-----|--------|-----------|
| Pesos del modelo | `models-v1` | `rdv-rtdetrv2-r18-rdd2022-v1.onnx` | Archivo ONNX fijado que descarga `rdv models pull` y que se embebe en los bundles de escritorio |
| Aplicación | `v0.1.0` | wheel, sdist, tres bundles de escritorio, `SHA256SUMS`, atestaciones de procedencia | CLI (L1) y aplicación de escritorio portátil (L3) |

El release de los pesos se crea **una sola vez** (el asset v1 se publicó en T10) y el
workflow de tags nunca lo reconstruye. El workflow de tags lo descarga con
`gh release download`, verifica su sha256 contra `src/rdv/assets/models.json`, lo embebe
en cada bundle y reutiliza el mismo digest en el smoke posterior al release. Si los dos
digests no coinciden, el workflow se detiene antes de publicar nada.

## Antes del primer release

1. Confirma que el remoto es accesible y que CI está en verde en `main`.
2. Confirma que existe el release de los pesos. Se publica una vez por versión del modelo;
   vuelve a ejecutar el script solo si falta `models-v1` o si el asset está corrupto:

   ```bash
   bash scripts/publish_model.sh
   ```

   El script rechaza un artefacto cuyo sha256 o tamaño no coincide con el manifiesto. Sin
   este release fallan los jobs de bundles y `rdv models pull`.
3. Confirma que el manifiesto fija una entrada completa:

   ```bash
   uv run python scripts/ci/verify_release_inputs.py v0.1.0
   ```

## Orden del release

El primer tag `v0.1.0` se crea solo después de la primera ejecución de CI completamente en
verde sobre `main` y de que existan tanto el asset `models-v1` como el propio tag; el
workflow no puede arrancar en un runner cuya cuenta no pueda ejecutar Actions.

1. **Sube la versión** en `pyproject.toml` (y actualiza `uv.lock` solo si cambió una
   dependencia).
2. **Regenera el changelog** a partir de Conventional Commits:

   ```bash
   uvx git-cliff@2.14.2 --tag v0.1.0 -o CHANGELOG.md
   ```

   Revisa la sección y confírmala junto con el cambio de versión.
3. **Ejecuta la verificación local previa** (las mismas comprobaciones que ejecuta el
   workflow):

   ```bash
   uv lock --check
   uv run ruff check . && uv run ruff format --check .
   uv run mypy --strict src
   uv run lint-imports
   uv run pytest -q
   uv run python scripts/ci/verify_release_inputs.py v0.1.0
   uvx git-cliff@2.14.2 --tag v0.1.0 --unreleased > /dev/null
   ```

4. **Crea el tag y súbelo** como tag anotado. Subir el tag es lo que inicia
   `.github/workflows/release.yml`; el tag debe ser igual a `v` + la versión de
   `pyproject.toml`, o el job `verify` falla antes de construir.

   ```bash
   git tag -a v0.1.0 -m "road-damage-vision 0.1.0"
   git push origin v0.1.0
   ```

5. **Sigue el workflow**:

   ```bash
   gh run list --workflow release.yml --limit 3
   gh run watch <run-id> --exit-status
   ```

## Qué hace el workflow

| Job | Resultado |
|-----|-----------|
| `verify` | El tag es igual a la versión del proyecto; al menos un modelo fija un sha256 de 64 caracteres, tamaño, URL, umbral y licencia |
| `dist` | `uv build` produce la wheel y el sdist; `twine check` valida ambos |
| `bundles` | Matriz `ubuntu-24.04` → `linux-x64`, `windows-latest` → `windows-x64`, `macos-latest` → `macos-arm64`; descarga y verifica `models-v1`, construye el bundle onedir de PyInstaller, ejecuta `scripts/smoke_bundle.py` y prepara `RoadDamageVision-<version>-<plataforma>.{tar.gz,zip}` |
| `release` | Genera las notas con `git-cliff --latest --strip header`, calcula `SHA256SUMS`, atestigua la procedencia de cada asset y publica el release |
| `post-release-smoke` | Instala la wheel publicada con `uv tool install`, descarga el modelo real (`rdv models pull` verifica el sha256) y ejecuta `scripts/smoke_release.py` contra `reports/v1/sample_expectations.json` |

El smoke posterior al release exige que cada clase esperada esté presente con un puntaje
igual o superior al umbral fijado. El `sample_expectations.json` actual registra
`margin_ok: false` para las seis muestras porque la cabeza v1 tiene confianzas bajas; por
eso el smoke verifica la presencia de clases y el umbral, no el margen de 0.15 descrito en
`docs/MODEL_CARD.md`.

## Verificar una descarga

Los usuarios pueden verificar los bytes exactos que descargaron:

```bash
sha256sum -c SHA256SUMS
gh attestation verify road_damage_vision-0.1.0-py3-none-any.whl \
  --repo Eduardo12Pacheco/deteccion-fallas-viales
```

En Windows, usa `Get-FileHash -Algorithm SHA256 <archivo>` y compara con la línea
correspondiente de `SHA256SUMS`.

Los archivos de los bundles de escritorio **no están firmados**. macOS los bloquea con
Gatekeeper hasta que el usuario permita la aplicación en *Ajustes del Sistema → Privacidad
y seguridad*, y Windows muestra SmartScreen hasta que el usuario elija *Más información →
Ejecutar de todas formas*. El bundle de Linux requiere glibc 2.39 o posterior porque se
construye sobre Ubuntu 24.04.

## Rollback

- Nunca muevas ni reutilices un tag publicado: los releases son solo de adición.
- Si un asset está mal, publica una versión de parche (`v0.1.1`). Como medida de emergencia,
  un responsable puede eliminar el GitHub Release y su tag, pero el tag no debe recrearse;
  el siguiente intento recibe un nuevo número de versión.
- Si el asset de los pesos está corrupto, vuelve a subirlo solo si el sha256 del manifiesto
  no cambió (`scripts/publish_model.sh` es idempotente y verifica el digest). Un modelo nuevo
  necesita una nueva `version` en el manifiesto y un release `models-v2`, además de
  reconstruir los bundles.
- Si falla el smoke posterior al release, corrige hacia adelante con un release de parche; no
  edites `reports/v1/sample_expectations.json` para que el smoke pase (la decisión D17
  mantiene sin cambios los números publicados).

## Nota sobre el repositorio privado

El repositorio es privado, así que los assets de release requieren autenticación. `gh` usa
`GH_TOKEN`/`GITHUB_TOKEN` automáticamente. `rdv models pull` también acepta cualquiera de
las dos variables y resuelve el asset del release mediante la API de GitHub; sin token, la
descarga falla con un mensaje claro en lugar de un 404 anónimo. Por eso el smoke posterior
al release exporta `GITHUB_TOKEN` para el paso de descarga.

## Solución de problemas

- `tag vX does not match pyproject version Y`: sube la versión en `pyproject.toml`, confírmala,
  elimina el tag local con `git tag -d vX` y crea el tag sobre el commit corregido.
- `model asset ... does not match the manifest`: `models-v1` se reconstruyó o cambió el digest
  del manifiesto. Vuelve a ejecutar `scripts/publish_model.sh` desde el artefacto que coincide
  con el manifiesto versionado.
- Timeout del smoke del bundle: revisa el log del job que falla; el script muestra las últimas
  30 líneas del log del lanzador y aplica un timeout de 90 s para la verificación de salud.
- El job de release no puede atestiguar: comprueba que el workflow conserve
  `id-token: write` y `attestations: write`; ambos son obligatorios para
  `actions/attest-build-provenance`.
