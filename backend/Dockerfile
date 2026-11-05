FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY backend/*requirements.txt /app/
ARG INSTALL_RESEARCH=false
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN if [ "$INSTALL_RESEARCH" = "true" ]; then \
      pip install torch==2.8.0 --index-url "$TORCH_INDEX_URL" && \
      pip install -r /app/research-requirements.txt; \
    else pip install -r /app/requirements.txt; fi
RUN apt-get update && apt-get install -y --no-install-recommends git libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY backend/ /app/
COPY deployment/seed_repo.py /app/seed_repo.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
