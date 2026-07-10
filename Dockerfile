FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /opt/oncosentinel

COPY pyproject.toml requirements.txt ./
RUN pip install --no-deps -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY tests ./tests
COPY README.md LICENSE ./

RUN pip install --no-deps -e .

ENTRYPOINT ["fm-mapb"]
CMD ["--help"]
