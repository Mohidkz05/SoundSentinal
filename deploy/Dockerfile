# The model server, hosted. Built from the repository root by
# .github/workflows/model-image.yml and pushed to ghcr.io; run on Azure
# Container Apps by deploy/azure.sh. Code only — the weights are not in the
# image. They live on a private Azure file share mounted at $CKPT_ROOT.
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CKPT_ROOT=/mnt/checkpoints
WORKDIR /home/user/ai_model

COPY --chown=user deploy/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt

COPY --chown=user ai_model/app.py ai_model/model.py ai_model/aasist.py \
     ai_model/ssl_aasist.py ai_model/xlsr_300m_config.json ./

EXPOSE 8000
# app.py loads the model at import; waitress serves the same `app` object the
# dev server does. app.py's own __main__ binds loopback and is not used here.
CMD ["waitress-serve", "--host=0.0.0.0", "--port=8000", "--threads=2", "app:app"]
