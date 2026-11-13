FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/var/cache/huggingface \
    HF_HUB_CACHE=/var/cache/huggingface/hub

WORKDIR /app

# The lock is the Linux ARM64 CPU environment validated by CI. Its sole editable
# project line is filtered so pip cannot resolve build dependencies separately.
COPY requirements/arm64-cpu.lock /tmp/arm64-cpu.lock
RUN awk '{ sub(/\r$/, ""); if ($0 != "-e .") print }' /tmp/arm64-cpu.lock > /tmp/arm64-cpu-pinned.txt \
    && python -m pip install --no-cache-dir --requirement /tmp/arm64-cpu-pinned.txt \
    && rm /tmp/arm64-cpu.lock /tmp/arm64-cpu-pinned.txt

COPY src/ /app/src/
COPY scripts/create_user.py scripts/reset_owner_password.py /app/scripts/

RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid 10001 --home-dir /home/rag --create-home \
        --shell /usr/sbin/nologin rag \
    && install -d -o 10001 -g 10001 -m 0700 \
        /var/lib/rag-agent/documents /var/cache/huggingface /home/rag/.cache

USER 10001:10001
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "rag_agent.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "172.30.105.2"]
