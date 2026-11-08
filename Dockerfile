FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache

RUN useradd --create-home --uid 1000 app
WORKDIR /app

# FFmpeg comes from the imageio-ffmpeg wheel; no system packages needed.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app config.yaml transitions.yaml main.py ./
COPY --chown=app:app src ./src

USER app
CMD ["python", "main.py"]
