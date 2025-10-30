FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Dépendances pour le backend Python (tokenization)
RUN pip3 install --no-cache-dir \
    transformers==4.44.2 tokenizers==0.19.1 huggingface_hub==0.24.6 onnx==1.16.1

ENV TOKENIZERS_PARALLELISM=true

# Copie du repo de modèles + scripts
COPY models /models
COPY scripts /opt/scripts

# Télécharge les poids ONNX + fichiers tokenizer dans l'image
RUN bash /opt/scripts/download_assets.sh /models

EXPOSE 8000 8001 8002
CMD ["tritonserver","--model-repository=/models","--strict-model-config=true","--exit-on-error=true"]
