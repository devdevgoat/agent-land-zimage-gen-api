FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /srv
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY app ./app
COPY API.md ./API.md

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
