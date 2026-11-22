# Contribuir

Gracias por echarle un vistazo. Este es un proyecto de portafolio, pero se aceptan issues y pull
requests siempre que mantengan en verde las compuertas de calidad.

## Configuración de desarrollo

Requisitos: [uv](https://docs.astral.sh/uv/) y Git. uv aprovisiona Python 3.12.

```bash
uv sync --extra train --extra cpu    # full environment (training stack, CPU torch)
make ci-local                        # every CI job command, including clean-install smoke
```

La instalación base con `uv sync` (sin extras) basta para la inferencia, la CLI, la aplicación web y
la matriz de pruebas sin torch.

## Verificaciones

| Comando | Qué garantiza |
|---|---|
| `uv run ruff check .` / `uv run ruff format --check .` | Lint y formato |
| `uv run mypy --strict src` | Tipado estricto en cada archivo fuente |
| `uv run lint-imports` | Contratos de capas hexagonales, pureza e independencia de adaptadores |
| `uv run pytest -q` | Pruebas unitarias, de integración y end-to-end de la CLI y de la API |
| `uv run pytest -q -m ui` | Flujos de navegador con Playwright (solo localhost) |
| `uv run pytest -q --cov=src/rdv --cov-fail-under=85` | Compuerta de cobertura |
| `uv run pytest tests/unit/test_docs_links.py -q` | Cada enlace relativo y cada `#ancla` se resuelven |
| `uv lock --check` | El lockfile coincide con `pyproject.toml` |

Las pruebas son deterministas y sin conexión por defecto (`pytest-socket`); usa
`httpx.MockTransport` o fixtures versionados para los adaptadores de red. Marcadores: `train`
(requiere torch), `ui` (Playwright), `parity` (`pycocotools`) y `realmodel` (nunca en CI).

## Convenciones

- Conventional Commits en inglés, un cambio coherente por commit, sin atribuciones de IA.
- Test first: agrega la prueba que falla, luego la implementación y después refactoriza.
- Mantén la dirección de dependencias `entrypoints > composition > adapters > application > domain`;
  `tests/unit/test_architecture.py` lo verifica.
- La lógica pura va en `rdv.domain` (solo biblioteca estándar y numpy); la E/S de terceros va en un
  adaptador y se conecta en `composition.py`.
- No subas datasets, pesos de modelos, directorios de corridas, rutas absolutas ni secretos. Los
  fixtures y las muestras siguen la postura CC BY-SA 4.0 de [docs/DATA.md](docs/DATA.md).
- Actualiza el documento correspondiente cuando cambie el comportamiento, y registra una decisión
  significativa como un ADR nuevo en [docs/adr/](docs/adr/README.md).
- El código, los comentarios y las cadenas de la interfaz están en inglés. Las excepciones son
  `README.md` (en español) y las etiquetas de clase y las traducciones de la interfaz en español.

## Pull requests

Mantén el diff enfocado, describe el cambio de comportamiento y enumera los comandos que ejecutaste.
Un pull request está listo cuando `make ci-local` está en verde en tu máquina y los README y la
documentación siguen resolviendo sus enlaces. Los reportes de seguridad siguen el proceso descrito en
[SECURITY.md](SECURITY.md), no un issue público.
