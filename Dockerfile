# Runs on Render's free plan (which sets $PORT); defaults to port 7860 anywhere else.
FROM python:3.11-slim

# Run as a normal user, not root.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user . .
RUN pip install --no-cache-dir --user -e ".[serve]" \
 && python -c "from dc.embed import load_model; load_model()"

EXPOSE 7860
# One worker: the whole service has to fit in 512 MB.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1
