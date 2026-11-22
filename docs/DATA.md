# Datos: fuentes de RDD2022, licencias, checksums y linaje

Todo lo documentado aquí es reproducible con los comandos de este archivo. No se versionan bytes de
datos en git: el mini fixture y las seis muestras empaquetadas son las únicas imágenes redistribuidas
(ver [Fixtures y muestras](#fixtures-y-muestras)).

## Fuente

| Campo | Valor |
|---|---|
| Dataset | RDD2022 - The multi-national Road Damage Dataset, publicado a través de CRDDC'2022 |
| Artículo de Figshare | `21431547` (DOI `10.6084/m9.figshare.21431547.v1`) |
| URL de descarga | `https://ndownloader.figshare.com/files/38030910` |
| Archivo externo | `RDD2022_released_through_CRDDC2022.zip` |
| Tamaño externo | 13,264,172,619 bytes |
| md5 externo | `b62bd51d2ffcfaa76c60f234f0cc2bb3` |
| Paper | RDD2022: A multi-national image dataset for automatic road damage detection, arXiv:2209.08538 |
| Clases | `D00` grieta longitudinal, `D10` grieta transversal, `D20` grieta de piel de cocodrilo, `D40` bache |

El ZIP externo contiene un ZIP interno sin comprimir (stored) por país. El proyecto descarga solo los
archivos internos que necesita mediante peticiones HTTP con rangos; el archivo de Noruega de 10.6 GB se
muestrea mediante rangos anidados sin descargarlo completo. Ver `docs/adr/0009-ranged-nested-zip.md`.

### Archivos internos fijados

`data/rdd2022.lock.json` es la única fuente de verdad para los desplazamientos y los checksums.
`header_offset` es la posición en bytes del encabezado local del ZIP interno dentro del archivo externo;
el adaptador analiza ese encabezado para encontrar el primer byte de datos, de modo que un campo extra
Zip64 no pueda desplazar el rango en silencio.

| País | Miembro externo | Desplazamiento del encabezado | Tamaño (bytes) | CRC32 | sha256 |
|---|---|---:|---:|---|---|
| Japón | `RDD2022/Japan.zip` | 1,136,050,224 | 1,072,598,857 | `e133e2cc` | `76efe7acb4642da283891611670dac7e8866a1b1dd145b8a1540fd440b1328f7` |
| India | `RDD2022/India.zip` | 609,341,032 | 526,709,145 | `f186f6f8` | `28eab0aa85638855e1907e63b175be2305a0accd018c54603037d3fb8cab963a` |
| Chequia | `RDD2022/Czech.zip` | 352,223,767 | 257,117,218 | `abcf66aa` | `94c65a4e6cfed187b193ba967f0a2b16bd4287dfe313d0962f3711afce75bf1d` |
| Noruega (solo muestra OOD) | `RDD2022/Norway.zip` | 2,208,649,128 | 10,611,186,814 | `01bedb71` | no descargado (muestreado mediante rangos anidados) |
| United_States (excluido) | `RDD2022/United_States.zip` | 12,819,836,010 | 444,335,958 | `fb8ded4b` | no descargado |
| China_Drone (excluido) | `RDD2022/China_Drone.zip` | 0 | 160,193,541 | `28b57b73` | no descargado |
| China_MotorBike (excluido) | `RDD2022/China_MotorBike.zip` | 160,193,594 | 192,030,116 | `f887e542` | no descargado |

`rdv data download` escribe `<data_dir>/downloads/<Country>.zip.part`, reanuda un part truncado, verifica
el tamaño exacto y el CRC32, y solo entonces lo renombra a `<Country>.zip`. La primera descarga exitosa
registra el sha256 del archivo interno en el lock, y las ejecuciones posteriores lo verifican. Un
desajuste de CRC32 elimina el part y termina con código 3.

## Postura de licencia

- **Registro de Figshare:** CC BY 4.0 (el campo `license` de la API de Figshare).
- **Evidencia contradictoria:** la página del paper en arXiv y un mirror de Hugging Face indican
  CC BY-SA 4.0. Esa es la licencia del paper; el mirror pudo haberla copiado.
- **Decisión del repositorio (D2):** seguir el registro de Figshare para los datos, atribuir a los
  autores y redistribuir los fixtures/muestras derivados bajo CC BY-SA 4.0, lo que satisface ambas
  lecturas. La licencia MIT de este repositorio cubre solo el código.
- **Modelos entrenados con los datos:** la situación legal de los pesos derivados de imágenes con
  licencia CC no está resuelta; la model card lo declara explícitamente en lugar de afirmar una licencia
  limpia.
- **United_States / Google Street View:** excluido del entrenamiento y no redistribuido, porque las
  imágenes de Street View tienen términos separados que la licencia del dataset no puede resolver.
- **Anotaciones de test:** el paper de RDD2022 indica que las anotaciones de las imágenes de test no se
  publicaron. Las carpetas oficiales `test/` contienen solo imágenes, así que este proyecto evalúa sobre
  su propio split retenido (ver abajo) y nunca afirma puntajes sobre el test oficial del desafío.

## Descarga y estructura

```bash
export RDV_DATA_DIR=data          # keep ~2 GB of data out of git and ~/.local
uv run rdv data download --countries Japan India Czech
uv run rdv data verify
uv run rdv data stats
uv run rdv data prepare
# Optional out-of-domain sample (ranged reads into the 10.6 GB Norway ZIP):
uv run rdv data download --ood-norway 1000
```

El árbol resultante es:

```
<data_dir>/
├── downloads/                 # verified inner ZIPs (+ optional outer ZIP)
└── rdd2022/
    ├── splits.json            # rdv.splits/v1
    ├── Japan/train/{images,annotations/xmls}/
    ├── India/train/{images,annotations/xmls}/
    ├── Czech/train/{images,annotations/xmls}/
    └── Norway/train/{images,annotations/xmls}/   # 1,000-image OOD sample
```

`RDV_OFFLINE=1` hace que cada comando falle antes de la primera petición de red. `rdv data download --full`
descarga el ZIP externo de 13 GB, verifica su md5 y extrae localmente los archivos internos solicitados;
es el respaldo si alguna vez dejan de funcionar las peticiones por rangos de Figshare.

## Splits (D3)

Las carpetas de train etiquetadas se dividen por país con un split por bloques con semilla:

- los ids se ordenan y se cortan en bloques de 100 ids consecutivos;
- los bloques se mezclan con `random.Random(42)`;
- los bloques se asignan de forma voraz a los objetivos 80/10/10;
- el manifiesto combinado se escribe en `<data_dir>/rdd2022/splits.json` (esquema `rdv.splits/v1`) y su
  sha256 queda fijado en `data/splits.lock.json`.

| Split | Chequia | India | Japón | Imágenes |
|---|---:|---:|---:|---:|
| train | 2,229 | 6,106 | 8,406 | 16,741 |
| val | 300 | 800 | 1,100 | 2,200 |
| test | 300 | 800 | 1,000 | 2,100 |
| **total** | **2,829** | **7,706** | **10,506** | **21,041** |

sha256 del manifiesto: `dbc3eaed76cf2c85e7b3203ebd9248e66daa0caf5fe0b5c7319d76df917e0b2f`
(fijado en `data/splits.lock.json`).

`rdv data prepare` recalcula el manifiesto a partir de las imágenes descargadas (no a partir de un
manifiesto anterior) y rechaza un digest que no coincida con el lock. El lock solo se escribe si no
existe; borrarlo es la forma explícita de aceptar un split nuevo.

## Estadísticas medidas

Comando: `RDV_DATA_DIR=data uv run rdv data stats` (verificado el 2026-11-20 en la máquina del
responsable; los tres archivos coincidieron con su tamaño, CRC32 y sha256 fijados).

| País | Imágenes de train | Etiquetas | D00 | D10 | D20 | D40 | Descartadas |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chequia | 2,829 | 1,745 | 988 | 399 | 161 | 197 | 0 |
| India | 7,706 | 6,831 | 1,555 | 68 | 2,021 | 3,187 | 1,372 |
| Japón | 10,506 | 16,469 | 4,049 | 3,979 | 6,198 | 2,243 | 8,284 |
| Noruega (muestra OOD) | 1,000 | 1,193 | 921 | 185 | 45 | 42 | 0 |
| **Total** | **22,041** | **26,238** | **7,513** | **4,631** | **8,425** | **5,669** | **9,656** |

Los conteos de imágenes dentro de dominio coinciden exactamente con los conteos publicados de train por
país. La fila de Noruega es la muestra OOD de 1,000 imágenes (mezcla con semilla, nunca usada para
entrenar), leída mediante rangos anidados del archivo de 10.6 GB y etiquetada con el mismo parser VOC.
Las etiquetas fuera de las cuatro clases soportadas se descartan y se cuentan en lugar de mapearse en
silencio:

| Etiqueta original | Conteo | Nota |
|---|---:|---|
| `D44` | 5,057 | Subtipo adicional de grieta de RDD2022, fuera de las cuatro clases de este proyecto |
| `D50` | 3,581 | Subtipo adicional de RDD2022, fuera de las cuatro clases de este proyecto |
| `D43` | 793 | Subtipo adicional de RDD2022, fuera de las cuatro clases de este proyecto |
| `D01` | 179 | Variante de grieta longitudinal que no está en la lista de clases D4 |
| `D11` | 45 | Variante de grieta transversal que no está en la lista de clases D4 |
| `D0w0` | 1 | Etiqueta malformada en las anotaciones de origen |

El parser también descarta las cajas de ground truth que colapsan a área cero tras el recorte; esas se
cuentan internamente en el parser VOC, pero no forman parte de `rdv data stats`. Los XML vacíos se
conservan como imágenes negativas.

## Fixtures y muestras

`scripts/make_fixtures.py` implementa D15(b)(c) con semilla 42:

- 3 imágenes positivas por clase de Japón/India/Chequia (12 en total) más 4 negativos (imágenes cuyo XML
  no tiene objetos);
- JPEG redimensionado a 640 px en el lado largo (calidad 85);
- XML VOC reescrito con las nuevas dimensiones y las cajas escaladas, una anotación por cada caja de
  ground truth que sobrevive;
- `tests/fixtures/rdd2022-mini/splits.json` usa un split por bloques de 4 imágenes (proporciones
  0.5/0.25/0.25) para que las tres particiones no estén vacías;
- seis muestras se copian desde el fixture (la primera de cada clase, la segunda de D40, la segunda de
  D00) a `src/rdv/assets/samples/`;
- `data/fixtures.lock.json` registra el sha256 de cada archivo producido.

Ambos directorios incluyen `LICENSE` (aviso CC BY-SA 4.0) y `ATTRIBUTION.md` (ids de las imágenes de
origen, DOI, paper y discusión de la licencia). El mini fixture está pensado solo para tests, docs y las
muestras incluidas; para entrenar, usa el dataset completo.

Regenera y verifica:

```bash
uv run python scripts/make_fixtures.py
git diff --stat          # must be empty for byte-identical fixtures
du -sh tests/fixtures/rdd2022-mini
```

## Linaje

```
Figshare outer ZIP (pinned md5/size)
  └─ ranged inner ZIP Japan/India/Czech (pinned CRC32, recorded sha256)
       └─ extract <Country>/train/{images,annotations/xmls}
            ├─ seeded 80/10/10 block split → splits.json (sha256 pinned)
            ├─ VOC parse → rdv data stats (counts, dropped labels)
            └─ training / evaluation (T9, T10)
  └─ ranged nested Norway ZIP → 1,000 seeded train images (OOD only)
  └─ scripts/make_fixtures.py → mini fixture + samples + fixtures.lock.json
```

Las etiquetas fuera de `D00/D10/D20/D40` se descartan y se cuentan en el parser; `rdv data stats` informa
el texto original de la etiqueta, de modo que un cambio en la fuente sea visible. Las imágenes cuyo XML no
tiene objetos se conservan como negativos, no se descartan.
