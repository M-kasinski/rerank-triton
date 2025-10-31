# rerank-triton

Ce dépôt fournit un dépôt de modèles Triton complet pour servir le modèle de rerank "mxbai-rerank-xsmall-v1" via un pipeline pré-traitement Python → ONNX Runtime → post-traitement encapsulé dans un modèle ensemble.

## Structure

```
.
├── Dockerfile
├── models/
│   ├── mxbai_xs_preproc/
│   ├── mxbai_xs_ort/
│   ├── mxbai_xs_postproc/
│   └── mxbai_xs_ensemble/
└── scripts/
    └── download_assets.sh
```

- `scripts/download_assets.sh` télécharge les poids ONNX et les fichiers du tokenizer depuis Hugging Face et les place dans l'arborescence.
- `models/mxbai_xs_preproc` contient un backend Python qui tokenise une requête et une liste de documents.
- `models/mxbai_xs_ort` embarque le modèle ONNX d'origine, exécuté via ONNX Runtime sur GPU (CUDA).
- `models/mxbai_xs_postproc` applique une sigmoïde sur les logits pour retourner des scores normalisés.
- `models/mxbai_xs_ensemble` chaîne ces étapes pour exposer un unique point d'entrée.

## Construction de l'image Docker

Assurez-vous d'avoir Docker installé ainsi que l'accès au registre `nvcr.io` (NVIDIA) pour récupérer l'image Triton de base. Cette configuration suppose que votre machine (ou VM) expose au moins un GPU NVIDIA accessible par Docker. Avant de construire l'image, téléchargez les artefacts Hugging Face :

```bash
scripts/download_assets.sh          # télécharge tokenizer + ONNX dans models/
docker build -t mxbai-triton:xs .
```

## Exécution du serveur Triton

```bash
docker run --rm \
  --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  mxbai-triton:xs
```

> ℹ️ ONNX Runtime est configuré pour utiliser le GPU (instance group `KIND_GPU`). Prévoyez une cible avec CUDA installé ; à défaut, changez `models/mxbai_xs_ort/config.pbtxt` pour basculer sur CPU (`KIND_CPU`).

## Vérifications et requêtes d'inférence

Une fois le serveur démarré, vous pouvez valider l'état du service et déclencher une requête d'inférence JSON.

```bash
curl -s localhost:8000/v2/health/ready
curl -s localhost:8000/v2/models/mxbai_xs_ensemble

curl -s -X POST localhost:8000/v2/models/mxbai_xs_ensemble/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs":[
      {"name":"QUERY","datatype":"BYTES","shape":[1,1],"data":["python async http client"]},
      {"name":"DOCUMENTS","datatype":"BYTES","shape":[1,3],
       "data":["doc A ...","doc B ...","doc C ..."]},
      {"name":"TOP_N","datatype":"INT32","shape":[1,1],"data":[2]}
    ],
    "outputs":[{"name":"scores"},{"name":"indices"}]
  }' | jq .
```

Les scores renvoyés correspondent à la probabilité normalisée (sigmoïde). Le tenseur `indices` contient les positions d'origine des documents triés par score décroissant (ici, les deux meilleurs). Les tensors de sortie incluent une première dimension de taille 1 (batch implicite de Triton) : `scores.shape = [1, TOP_N]`, `indices.shape = [1, TOP_N]`. Il suffit de sélectionner l'axe zéro pour retrouver la liste plate.

### Batching & perfs

- Le dépôt est configuré avec `max_batch_size = 512` sur l'ensemble du pipeline. Chaque requête peut contenir jusqu'à 512 documents, et Triton peut regrouper plusieurs requêtes simultanées (le premier axe de chaque entrée correspond au batching implicite de Triton).
- Le modèle ONNX s'exécute sur GPU (CUDA) via Triton ; pré/post-traitement restent en Python sur CPU. Le backend ONNX Runtime est configuré avec l'accélérateur `cuda` et des tailles de lot préférées jusqu'à 512 documents. Ajustez ces valeurs selon vos scénarios de charge pour limiter la latence.

## Publication continue sur GHCR

Un workflow GitHub Actions (`.github/workflows/docker-image.yml`) construit l'image Docker à chaque `push` (et sur déclenchement manuel). Lors d'un push sur une branche, l'image est taggée automatiquement avec :

- le nom de la branche (`feature-x` → `ghcr.io/<repo>:feature-x`),
- le couple branche + hash court (`ghcr.io/<repo>:feature-x-<sha>`),
- `latest` lorsque la branche est `main`.

Si un fichier `VERSION` est présent à la racine du dépôt, sa valeur est également injectée dans les tags pour la branche courante (`feature-x-1.2.3`). Sur `main`, le tag `1.2.3` est publié en plus de `latest`.

Les images ne sont poussées que pour les événements `push`. Les caches de compilation sont mutualisés grâce à `cache-from/cache-to` afin d'accélérer les builds successifs.
