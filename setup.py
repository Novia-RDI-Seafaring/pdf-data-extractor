from setuptools import setup, find_packages

setup(
    name='pdf_data_extractor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'aiofiles==23.2.1',
        'aiohttp==3.9.3',
        'aiosignal==1.3.1',
        'altair==5.2.0',
        'annotated-types==0.6.0',
        'anyio==3.7.1',
        'asttokens==2.4.1',
        'asyncer==0.0.2',
        'attrs==23.2.0',
        'backoff==2.2.1',
        'beautifulsoup4==4.12.3',
        'bidict==0.23.1',
        'bleach==6.1.0',
        'certifi==2024.2.2',
        'charset-normalizer==3.3.2',
        'click==8.1.7',
        'colorama==0.4.6',
        'comm==0.2.1',
        'contourpy==1.2.0',
        'cycler==0.12.1',
        'dataclasses-json==0.5.14',
        'debugpy==1.8.1',
        'decorator==5.1.1',
        'defusedxml==0.7.1',
        'Deprecated==1.2.14',
        'dirtyjson==1.0.8',
        'distro==1.9.0',
        'executing==2.0.1',
        'fastapi==0.108.0',
        'fastapi-socketio==0.0.10',
        'fastjsonschema==2.19.1',
        'ffmpy==0.3.2',
        'filelock==3.13.1',
        'filetype==1.2.0',
        'fonttools==4.49.0',
        'frozenlist==1.4.1',
        'fsspec==2024.2.0',
        'googleapis-common-protos==1.62.0',
        'gradio==4.19.1',
        'gradio_client==0.10.0',
        'greenlet==3.0.3',
        'grpcio==1.60.1',
        'h11==0.14.0',
        'httpcore==0.17.3',
        'httpx==0.24.1',
        'huggingface-hub==0.20.3',
        'idna==3.6',
        'importlib-metadata==6.11.0',
        'importlib-resources==6.1.1',
        'ipykernel==6.29.2',
        'ipython==8.18.1',
        'jedi==0.19.1',
        'Jinja2==3.1.3',
        'joblib==1.3.2',
        'jsonpatch==1.33',
        'jsonpath-ng==1.6.1',
        'jsonpointer==2.4',
        'jsonschema==4.21.1',
        'jsonschema-specifications==2023.12.1',
        'jupyter_client==8.6.0',
        'jupyter_core==5.7.1',
        'jupyterlab_pygments==0.3.0',
        'kiwisolver==1.4.5',
        'langchain==0.1.8',
        'langchain-community==0.0.21',
        'langchain-core==0.1.25',
        'langsmith==0.1.5',
        'Lazify==0.4.0',
        'literalai==0.0.103',
        'llama-index==0.9.48',
        'markdown-it-py==3.0.0',
        'MarkupSafe==2.1.5',
        'marshmallow==3.20.2',
        'matplotlib==3.8.3',
        'matplotlib-inline==0.1.6',
        'mdurl==0.1.2',
        'mistune==3.0.2',
        'mpmath==1.3.0',
        'multidict==6.0.5',
        'mypy-extensions==1.0.0',
        'nbclient==0.9.0',
        'nbconvert==7.16.1',
        'nbformat==5.9.2',
        'nest-asyncio==1.6.0',
        'networkx==3.2.1',
        'nltk==3.8.1',
        'numpy==1.26.4',
        'openai==1.12.0',
        'opencv-python==4.9.0.80',
        'opentelemetry-api==1.22.0',
        'opentelemetry-exporter-otlp==1.22.0',
        'opentelemetry-exporter-otlp-proto-common==1.22.0',
        'opentelemetry-exporter-otlp-proto-grpc==1.22.0',
        'opentelemetry-exporter-otlp-proto-http==1.22.0',
        'opentelemetry-instrumentation==0.43b0',
        'opentelemetry-proto==1.22.0',
        'opentelemetry-sdk==1.22.0',
        'opentelemetry-semantic-conventions==0.43b0',
        'orjson==3.9.14',
        'packaging==23.2',
        'pandas==2.2.0',
        'pandocfilters==1.5.1',
        'parso==0.8.3',
        'pillow==10.2.0',
        'platformdirs==4.2.0',
        'ply==3.11',
        'prompt-toolkit==3.0.43',
        'protobuf==4.25.3',
        'psutil==5.9.8',
        'pure-eval==0.2.2',
        'pydantic==2.6.1',
        'pydantic-settings==2.2.1',
        'pydantic_core==2.16.2',
        'pydub==0.25.1',
        'Pygments==2.17.2',
        'PyJWT==2.8.0',
        'PyMuPDF==1.23.24',
        'PyMuPDFb==1.23.22',
        'pyparsing==3.1.1',
        'PyPDF2==3.0.1',
        'pypdfium2==4.27.0',
        'python-dateutil==2.8.2',
        'python-dotenv==1.0.1',
        'python-engineio==4.9.0',
        'python-graphql-client==0.4.3',
        'python-multipart==0.0.9',
        'python-socketio==5.11.1',
        'pytz==2024.1',
        #'pywin32==306',
        'PyYAML==6.0.1',
        'pyzmq==25.1.2',
        'referencing==0.33.0',
        'regex==2023.12.25',
        'requests==2.31.0',
        'rich==13.7.0',
        'rpds-py==0.18.0',
        'ruff==0.2.2',
        'safetensors==0.4.2',
        'semantic-version==2.10.0',
        'setuptools==69.1.0',
        'shellingham==1.5.4',
        'simple-websocket==1.0.0',
        'six==1.16.0',
        'sniffio==1.3.0',
        'soupsieve==2.5',
        'SQLAlchemy==2.0.27',
        'stack-data==0.6.3',
        'starlette==0.32.0.post1',
        'surya-ocr==0.2.2',
        'sympy==1.12',
        'syncer==2.0.3',
        'tabulate==0.9.0',
        'tenacity==8.2.3',
        'tiktoken==0.6.0',
        'tinycss2==1.2.1',
        'tokenizers==0.15.2',
        'tomli==2.0.1',
        'tomlkit==0.12.0',
        'toolz==0.12.1',
        'tornado==6.4',
        'tqdm==4.66.2',
        'traitlets==5.14.1',
        'transformers==4.36.2',
        'typer==0.9.0',
        'typing-inspect==0.9.0',
        'typing_extensions==4.9.0',
        'tzdata==2024.1',
        'uptrace==1.22.0',
        'urllib3==2.2.1',
        'uvicorn==0.25.0',
        'watchfiles==0.20.0',
        'wcwidth==0.2.13',
        'webencodings==0.5.1',
        'websockets==11.0.3',
        'wrapt==1.16.0',
        'wsproto==1.2.0',
        'yarl==1.9.4',
        'zipp==3.17.0',
        'pdfplumber'
    ],
)
