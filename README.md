# rerank-triton

Ce dépôt fournit un dépôt de modèles Triton complet pour servir le modèle de rerank "mxbai-rerank-xsmall-v1" via un pipeline pré-traitement → ONNX → post-traitement encapsulé dans un modèle ensemble.

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
- `models/mxbai_xs_ort` pointe vers le modèle ONNX pour calculer les logits.
- `models/mxbai_xs_postproc` applique une sigmoïde sur les logits pour retourner des scores normalisés.
- `models/mxbai_xs_ensemble` chaîne ces trois étapes pour exposer un unique point d'entrée.

## Construction de l'image Docker

Assurez-vous d'avoir Docker installé ainsi que l'accès au registre `nvcr.io` (NVIDIA) pour récupérer l'image Triton de base. La construction télécharge automatiquement les actifs nécessaires.

```bash
docker build -t mxbai-triton:xs .
```

## Exécution du serveur Triton

Le modèle ONNX est configuré pour tourner sur GPU (CUDA). Assurez-vous de monter une carte compatible (A10, L40S, H100...) et de passer l'accès GPU au conteneur, faute de quoi Triton refusera de charger le modèle.

```bash
docker run --rm \
  --gpus all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  mxbai-triton:xs
```

> ℹ️ Les étapes de pré/post-traitement restent en Python sur CPU, seule l'inférence ONNX consomme le GPU.

## Vérifications et requêtes d'inférence

Une fois le serveur démarré, vous pouvez valider l'état du service et déclencher une requête d'inférence JSON.

```bash
curl -s localhost:8000/v2/health/ready
curl -s localhost:8000/v2/models/mxbai_xs_ensemble

curl -s -X POST localhost:8000/v2/models/mxbai_xs_ensemble/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs":[
      {"name":"QUERY","datatype":"STRING","shape":[1],"data":["python async http client"]},
      {"name":"DOCUMENTS","datatype":"STRING","shape":[3],
       "data":["doc A ...","doc B ...","doc C ..."]},
      {"name":"TOP_N","datatype":"INT32","shape":[1],"data":[2]}
    ],
    "outputs":[{"name":"scores"},{"name":"indices"}]
  }' | jq .
```

Les scores renvoyés correspondent à la probabilité normalisée (sigmoïde). Le tenseur `indices` contient les positions d'origine des documents triés par score décroissant (ici, les deux meilleurs).

## Publication continue sur GHCR

Un workflow GitHub Actions (`.github/workflows/docker-image.yml`) construit l'image Docker à chaque `push` (et sur déclenchement manuel). Lors d'un push sur une branche, l'image est taggée automatiquement avec :

- le nom de la branche (`feature-x` → `ghcr.io/<repo>:feature-x`),
- le couple branche + hash court (`ghcr.io/<repo>:feature-x-<sha>`),
- `latest` lorsque la branche est `main`.

Si un fichier `VERSION` est présent à la racine du dépôt, sa valeur est également injectée dans les tags pour la branche courante (`feature-x-1.2.3`). Sur `main`, le tag `1.2.3` est publié en plus de `latest`.

Les images ne sont poussées que pour les événements `push`. Les caches de compilation sont mutualisés grâce à `cache-from/cache-to` afin d'accélérer les builds successifs.
