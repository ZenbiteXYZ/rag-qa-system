FROM python:3.12.15-slim

# Server version matches the mlflow-tracing client family (3.17, see uv.lock).
RUN pip install --no-cache-dir mlflow==3.17.1 psycopg2-binary==2.9.13

RUN useradd --create-home --uid 10001 mlflow \
    && mkdir -p /mlflow/artifacts \
    && chown -R mlflow:mlflow /mlflow
USER mlflow
EXPOSE 5000
