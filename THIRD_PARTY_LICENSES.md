# Third-party licenses

`road-damage-vision` is licensed under the MIT License (see `LICENSE`). This
file lists the licenses of the Python distributions in the project's runtime
dependency closure: the base dependencies plus the `train` and `cpu` extras,
installed with `uv sync --extra train --extra cpu`.

Development and build-only dependencies are not part of the shipped runtime
and are therefore not listed here. In particular, PyInstaller (build group) is
GPL-2.0 with the Bootloader Exception and is never imported by the application
(ADR 0007); the `security.yml` license guard asserts that no installed runtime
dependency declares GPL or AGPL.

The table was generated with `pip-licenses`:

```bash
uv run pip-licenses --format=markdown --with-urls --order=license \
  --packages $(uv run python -c "from tests.unit.test_license_guard import runtime_distributions; print(' '.join(d.name for d in runtime_distributions()))")
```

Bundled binary notices that do not appear in Python package metadata:

- **OpenCV wheels** bundle a build of FFmpeg and other third-party libraries.
  FFmpeg components are LGPL-2.1-or-later; the `opencv-python-headless`
  distribution itself is Apache-2.0. FFmpeg is dynamically linked and can be
  replaced by the user, which satisfies the LGPL relinking requirement.
- **ONNX Runtime** and **PyTorch** wheels embed their own third-party notices;
  they are redistributed unmodified and their upstream license files apply.
- **Dataset images** are not software: see `docs/DATA.md` for the RDD2022
  attribution and the CC BY-SA 4.0 stance for redistributed derivatives.

| Name                   | Version    | License                                                                                                              | URL                                                                                |
|------------------------|------------|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| protobuf               | 7.36.2     | 3-Clause BSD License                                                                                                 | https://developers.google.com/protocol-buffers/                                    |
| flatbuffers            | 25.12.19   | Apache Software License                                                                                              | https://google.github.io/flatbuffers/                                              |
| huggingface_hub        | 0.36.2     | Apache Software License                                                                                              | https://github.com/huggingface/huggingface_hub                                     |
| opencv-python-headless | 5.1.0.95   | Apache Software License                                                                                              | https://github.com/opencv/opencv-python                                            |
| requests               | 2.34.2     | Apache Software License                                                                                              | https://github.com/psf/requests                                                    |
| safetensors            | 0.8.0      | Apache Software License                                                                                              | https://github.com/huggingface/safetensors                                         |
| tokenizers             | 0.22.2     | Apache Software License                                                                                              | https://github.com/huggingface/tokenizers                                          |
| transformers           | 4.57.6     | Apache Software License                                                                                              | https://github.com/huggingface/transformers                                        |
| hf-xet                 | 1.8.0      | Apache-2.0                                                                                                           | https://github.com/huggingface/xet-core                                            |
| opentelemetry-api      | 1.45.2     | Apache-2.0                                                                                                           | https://github.com/open-telemetry/opentelemetry-python/tree/main/opentelemetry-api |
| python-multipart       | 0.0.34     | Apache-2.0                                                                                                           | https://github.com/Kludex/python-multipart                                         |
| tzdata                 | 2026.6     | Apache-2.0                                                                                                           | https://github.com/python/tzdata                                                   |
| torch                  | 2.15.0+cpu | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib | https://pytorch.org                                                                |
| regex                  | 2026.11.19 | Apache-2.0 AND CNRI-Python                                                                                           | https://github.com/mrabarnett/mrab-regex                                           |
| packaging              | 26.3       | Apache-2.0 OR BSD-2-Clause                                                                                           | https://github.com/pypa/packaging                                                  |
| torchvision            | 0.29.2+cpu | BSD                                                                                                                  | https://github.com/pytorch/vision                                                  |
| Jinja2                 | 3.1.6      | BSD License                                                                                                          | https://github.com/pallets/jinja/                                                  |
| cycler                 | 0.12.1     | BSD License                                                                                                          | https://matplotlib.org/cycler/                                                     |
| httpx                  | 0.28.1     | BSD License                                                                                                          | https://github.com/encode/httpx                                                    |
| kiwisolver             | 1.5.1      | BSD License                                                                                                          | https://github.com/nucleic/kiwi                                                    |
| mpmath                 | 1.3.0      | BSD License                                                                                                          | http://mpmath.org/                                                                 |
| scipy                  | 1.18.1     | BSD License                                                                                                          | https://scipy.org/                                                                 |
| sympy                  | 1.14.0     | BSD License                                                                                                          | https://sympy.org                                                                  |
| Pygments               | 2.21.0     | BSD-2-Clause                                                                                                         | https://pygments.org                                                               |
| MarkupSafe             | 3.0.4      | BSD-3-Clause                                                                                                         | https://github.com/pallets/markupsafe/                                             |
| click                  | 8.5.1      | BSD-3-Clause                                                                                                         | https://github.com/pallets/click/                                                  |
| contourpy              | 1.4.0      | BSD-3-Clause                                                                                                         | https://github.com/contourpy/contourpy                                             |
| fsspec                 | 2026.9.0   | BSD-3-Clause                                                                                                         | https://github.com/fsspec/filesystem_spec                                          |
| httpcore               | 1.0.9      | BSD-3-Clause                                                                                                         | https://www.encode.io/httpcore/                                                    |
| idna                   | 3.20       | BSD-3-Clause                                                                                                         | https://github.com/kjd/idna                                                        |
| networkx               | 3.7        | BSD-3-Clause                                                                                                         | https://networkx.org/                                                              |
| python-dateutil        | 4.0.0      | BSD-3-Clause                                                                                                         | https://github.com/dateutil/dateutil                                               |
| python-dotenv          | 1.2.4      | BSD-3-Clause                                                                                                         | https://github.com/theskumar/python-dotenv                                         |
| starlette              | 1.10.0     | BSD-3-Clause                                                                                                         | https://github.com/Kludex/starlette                                                |
| uvicorn                | 0.54.0     | BSD-3-Clause                                                                                                         | https://uvicorn.dev/                                                               |
| numpy                  | 2.5.5      | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0                                                                   | https://numpy.org                                                                  |
| shellingham            | 1.5.4      | ISC License (ISCL)                                                                                                   | https://github.com/sarugaku/shellingham                                            |
| annotated-doc          | 0.0.7      | MIT                                                                                                                  | https://github.com/fastapi/annotated-doc                                           |
| annotated-types        | 0.8.0      | MIT                                                                                                                  | https://github.com/annotated-types/annotated-types                                 |
| anyio                  | 4.16.0     | MIT                                                                                                                  | https://anyio.readthedocs.io/en/stable/versionhistory.html                         |
| charset-normalizer     | 3.5.2      | MIT                                                                                                                  | https://github.com/jawah/charset_normalizer/blob/master/CHANGELOG.md               |
| fastapi                | 0.165.0    | MIT                                                                                                                  | https://github.com/fastapi/fastapi                                                 |
| filelock               | 4.2.12     | MIT                                                                                                                  | https://github.com/tox-dev/filelock                                                |
| fonttools              | 4.68.0     | MIT                                                                                                                  | http://github.com/fonttools/fonttools                                              |
| platformdirs           | 4.14.7     | MIT                                                                                                                  | https://github.com/tox-dev/platformdirs                                            |
| pydantic               | 2.14.4     | MIT                                                                                                                  | https://github.com/pydantic/pydantic                                               |
| pydantic-settings      | 2.16.3     | MIT                                                                                                                  | https://github.com/pydantic/pydantic-settings                                      |
| pydantic_core          | 2.50.5     | MIT                                                                                                                  | https://github.com/pydantic                                                        |
| pyparsing              | 3.3.3      | MIT                                                                                                                  | https://github.com/pyparsing/pyparsing/                                            |
| typer                  | 0.29.2     | MIT                                                                                                                  | https://github.com/fastapi/typer                                                   |
| typing-inspection      | 0.4.5      | MIT                                                                                                                  | https://github.com/pydantic/typing-inspection                                      |
| urllib3                | 2.8.0      | MIT                                                                                                                  | https://github.com/urllib3/urllib3/blob/main/CHANGES.rst                           |
| PyYAML                 | 6.0.3      | MIT License                                                                                                          | https://pyyaml.org/                                                                |
| h11                    | 0.16.0     | MIT License                                                                                                          | https://github.com/python-hyper/h11                                                |
| markdown-it-py         | 4.3.0      | MIT License                                                                                                          | https://github.com/executablebooks/markdown-it-py                                  |
| mdurl                  | 0.1.2      | MIT License                                                                                                          | https://github.com/executablebooks/mdurl                                           |
| onnxruntime            | 1.32.0     | MIT License                                                                                                          | https://onnxruntime.ai                                                             |
| rich                   | 15.0.0     | MIT License                                                                                                          | https://github.com/Textualize/rich                                                 |
| structlog              | 26.1.0     | MIT OR Apache-2.0                                                                                                    | https://github.com/hynek/structlog/blob/main/CHANGELOG.md                          |
| pillow                 | 13.0.0     | MIT-CMU                                                                                                              | https://python-pillow.org                                                          |
| tqdm                   | 4.70.1     | MPL-2.0 AND MIT                                                                                                      | https://tqdm.github.io                                                             |
| certifi                | 2026.10.23 | Mozilla Public License 2.0 (MPL 2.0)                                                                                 | https://github.com/certifi/python-certifi                                          |
| typing_extensions      | 4.16.0     | PSF-2.0                                                                                                              | https://github.com/python/typing_extensions                                        |
| matplotlib             | 3.11.2     | Python Software Foundation License                                                                                   | https://matplotlib.org                                                             |
