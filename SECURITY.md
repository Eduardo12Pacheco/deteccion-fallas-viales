# Política de seguridad

## Versiones soportadas

| Versión | Soportada |
|---|---|
| `0.1.x` (último release) | Sí |
| Releases anteriores | No |

Las correcciones de seguridad se publican como una nueva versión de parche; el repositorio es solo de
adición y los tags publicados nunca se mueven.

## Reportar una vulnerabilidad

Por favor, no abras un issue público. Usa el reporte privado de vulnerabilidades de GitHub en este
repositorio (pestaña "Security", opción "Report a vulnerability") e incluye:

- la versión o el commit afectado,
- una reproducción mínima,
- el impacto que observaste.

Puedes esperar un acuse de recibo en pocos días. Concede una ventana razonable para la corrección antes
de cualquier divulgación pública.

## Alcance

El proyecto se ejecuta localmente y no recopila telemetría. No requiere claves de API ni cuentas;
`rdv models pull` y `rdv data download` son los únicos comandos que hacen peticiones de red, y
`RDV_OFFLINE=1` las desactiva. Áreas que vale la pena reportar:

- el servidor HTTP local: validación de `Host` y `Origin`, límites de tamaño de carga, validación de
  MIME y de decodificación, limpieza de archivos temporales y manejo de rutas;
- el análisis de archivos: el lector de ZIP anidados por rangos, la verificación CRC32/sha256 y las
  descargas reanudables;
- la carga de modelos: validación del manifiesto, verificación de digests y análisis de ONNX;
- problemas de la cadena de suministro de dependencias o de build.

El servidor solo se enlaza a `127.0.0.1`, rechaza encabezados `Host` ajenos (421), rechaza las
peticiones de origen cruzado que modifican estado (403) y nunca deriva una ruta del sistema de archivos
a partir de un nombre de archivo enviado por el usuario. Consulta [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
y [docs/RELEASING.md](docs/RELEASING.md) para el comportamiento previsto.

## Fuera de alcance

- La precisión, los falsos positivos y negativos, o la idoneidad del modelo para un fin. La herramienta no
  es de seguridad crítica y no debe ser la única base para decisiones de mantenimiento vial, seguridad
  vial, seguros o asuntos legales.
- Las advertencias de los binarios de escritorio sin firmar de macOS y Windows (Gatekeeper y SmartScreen).
  Los builds no están firmados a propósito y se documentan en [docs/RELEASING.md](docs/RELEASING.md).
- Las vulnerabilidades de herramientas locales opcionales fuera del lockfile versionado.

## Cadena de suministro

La CI ejecuta gitleaks sobre todo el historial, `pip-audit` sobre la exportación bloqueada del runtime y
una guarda de licencias que rechaza dependencias de runtime AGPL/GPL. Los artefactos de build se
atestiguan con la procedencia de GitHub en los releases etiquetados. No se guardan secretos en el
repositorio; los workflows solo usan `GITHUB_TOKEN`.
