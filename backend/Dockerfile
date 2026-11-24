# ---- build stage: install dependencies into an isolated prefix ----
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ---- runtime stage: slim image, non-root user ----
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DATABASE_URL=sqlite:////data/app.db \
    PORT=8000
RUN useradd --create-home --uid 10001 app && mkdir /data && chown app:app /data
COPY --from=builder /install /usr/local
WORKDIR /srv
COPY app ./app
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT','8000'))"
# Shell form so $PORT (set by hosts such as Render/Railway) is honoured.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
