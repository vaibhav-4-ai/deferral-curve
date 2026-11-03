# One image for both hosts: Hugging Face Spaces (port 7860) and Render (sets $PORT).
FROM python:3.11-slim

# Hugging Face Spaces run containers as user 1000.
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
