FROM python:3.11-slim AS builder

WORKDIR /app

COPY requirements-runtime.txt .
RUN pip install --upgrade pip setuptools wheel && \
    grep -v '^xgboost==' requirements-runtime.txt > requirements-base.txt && \
    test -s requirements-base.txt || (echo "ERROR: requirements-base.txt is empty - check requirements-runtime.txt encoding (must be UTF-8, not UTF-16)" && exit 1) && \
    pip install --no-cache-dir --user --only-binary :all: -r requirements-base.txt && \
    pip install --no-cache-dir --user --only-binary :all: --no-deps xgboost==3.2.0

FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
