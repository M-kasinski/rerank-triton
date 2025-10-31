# Journal d’intégration TensorRT (31 octobre 2025)

## Résumé des interventions
- Désactivation du workflow CI GitHub (`.github/workflows/docker-image.yml`) pour éviter les builds automatiques lors des prochains push.
- Lancement de plusieurs jobs OVHcloud AI Training (`ovhai job run`) pour générer un moteur TensorRT FP16 à partir de `model.onnx`. Les premières tentatives ont échoué à cause :
  - de dépendances manquantes (`huggingface_hub`) dans l’image TensorRT officielle,
  - de timeouts HTTP lors du téléchargement Hugging Face,
  - de droits d’exécution sur `scripts/build_trt_plan.sh`,
  - de l’absence préalable du dossier `models/mxbai_xs_trt/1`,
  - de l’option `--workspace` dépréciée avec TensorRT 10.3 (remplacée par `--memPoolSize=workspace:4096MiB`),
  - et d’un montage en lecture seule qui empêchait la persistance du plan.
- Job concluant : `build-trt-plan-10` (ID `7ab90235-efd3-45d0-8f4b-f79c8911b4f2`) exécuté avec le volume `standalone` en lecture/écriture. Le moteur FP16 `model.plan` a été généré et sauvegardé dans `/workspace/rerank-triton/models/mxbai_xs_trt/1/` sur ce volume.
- Tentatives supplémentaires en cours (`build-trt-plan-upload`) pour regénérer le plan et le pousser automatiquement vers un bucket. Les jobs `05c6c07d`, `ca6c7e3e`, `5c7ad872`, `68ca422d` ont échoué respectivement pour : installation pip bloquée sur `/workspace/.local` puis `/root/.local`, et crash du client `ovhai job logs` lors de la récupération du détail d’un job `exit code 128`.

## État actuel
- Le dépôt local reflète toutes les modifications de configuration (pipeline ONNX→TensorRT, scripts, README, Dockerfile).
- Le moteur TensorRT n’a pas encore été rapatrié sur la machine locale ni versionné : il se trouve sur le volume `standalone` dans le chemin mentionné ci-dessus. *(obsolète, voir mise à jour du 31 octobre 2025 à 15h55 UTC-4)*
- Aucune copie n’a été poussée vers l’Object Storage `rerank-triton-artifacts@GRA` (le job de copie a échoué avant d’être relancé). Le nouveau pipeline « upload direct » est encore en cours de stabilisation. *(obsolète, voir mise à jour du 31 octobre 2025 à 15h55 UTC-4)*
- Les assets HF ont été téléchargés localement (`models/mxbai_xs_ort/1/model.onnx`, `models/mxbai_xs_preproc/1/assets/`) et devront être nettoyés si on ne souhaite pas les committer. *(nettoyés depuis la mise à jour du 31 octobre 2025 à 15h55 UTC-4)*
- La CLI `ovhai` plante désormais systématiquement sur ce poste (panic `system-configuration` lors de la détection du proxy) ce qui bloque les tentatives de récupération du plan sans solution de contournement.

## Mise à jour du 31 octobre 2025 à 15h55 UTC-4
- Job GPU `build-trt-plan-export4` (ID `7f851482-e6bd-43a4-a726-706e2234e644`) : dépendances installées dans `/tmp/pip`, cache Hugging Face redirigé vers `/tmp/hf_home`, repo cloné dans `/tmp/rerank-triton`. Le moteur TensorRT FP16 a été généré via `trtexec` puis copié vers le bucket `rerank-triton-artifacts@GRA/model.plan`.
- Téléchargement local réussi avec `ovhai bucket object download rerank-triton-artifacts@GRA model.plan -o models/mxbai_xs_trt/1/`. Le fichier `models/mxbai_xs_trt/1/model.plan` (~144 MiB) est maintenant présent dans le dépôt.
- Jobs préalables `build-trt-plan-export2` (ID `20f1a31c-ef8e-46ca-bafe-3f425ec925da`) et `build-trt-plan-export3` (ID `6b01d5c2-63a3-4715-88b9-bb4b7431c969`) ont échoué respectivement sur la création du worktree `/workspace` (volume en lecture seule) et sur l’écriture du cache `/workspace/.cache`. Les logs confirment que rien n’a été poussé vers le bucket lors de ces essais.
- À suivre : valider le moteur dans Triton, décider du sort des assets téléchargés par `scripts/download_assets.sh`, et purger les jobs OVH obsolètes pour clarifier l’historique.

### Mise à jour du 31 octobre 2025 à 17h10 UTC-4
- Tests Triton HTTP/gRPC sur le job `triton-ssh` (`a73355e1-70f0-421b-a25b-05a3d21ff266`) : même avec payload JSON encodé en base64 et client officiel `tritonclient[http]`, le backend Python échoue systématiquement dans `deserialize_bytes_tensor` (`pybind11::error_already_set`, buffer attendu >3 Go). Les requêtes sont closes côté serveur avant réponse.
- Décision : abandonner la voie TensorRT pour l’instant et revenir à la pile ONNX classique. Nettoyage des artefacts (`models/mxbai_xs_trt/`, `scripts/build_trt_plan.sh`, captures de test) et restauration des configs (sortie preprocess en `INT64`, logits ONNX en `FP32`, ensemble pointant sur `mxbai_xs_ort` avec exécution GPU via ONNX Runtime).
- À prévoir si l’on retente TensorRT : reproduire le bug avec un repo minimal et ouvrir un ticket NVIDIA, ou basculer sur un proxy gRPC pour contourner `deserialize_bytes_tensor`.

### Session SSH OVH (31 octobre 2025 à 15h35 UTC-4)
- Job `triton-ssh` (ID `a73355e1-70f0-421b-a25b-05a3d21ff266`) lancé dans l’image `nvcr.io/nvidia/tritonserver:24.08-py3` avec `--ssh-public-keys ~/.ssh/id_ed25519.pub` et montage du bucket `rerank-triton-artifacts@GRA:/mnt/artifacts:ro`.
- Connexion :
  ```bash
  ssh -i ~/.ssh/id_ed25519 a73355e1-70f0-421b-a25b-05a3d21ff266@gra.ai.cloud.ovh.net
  ```
- À l’intérieur :
  1. `git clone --branch codex/set-up-triton-server-and-update-readme https://github.com/M-kasinski/rerank-triton.git /workspace/rerank-triton`
  2. `python3 -m pip install --no-cache-dir --target /tmp/pip …` (packages HF), puis `PYTHONPATH=/tmp/pip scripts/download_assets.sh`
  3. `cp /mnt/artifacts/model.plan /workspace/rerank-triton/models/mxbai_xs_trt/1/model.plan`
  4. Lancer Triton :  
     ```bash
     export PYTHONPATH=/tmp/pip;$ 
     export HF_HOME=/tmp/hf_home;$ 
     nohup /opt/tritonserver/bin/tritonserver \
       --model-repository=/workspace/rerank-triton/models \
       --http-port=8000 --grpc-port=8001 --metrics-port=8002 \
       --exit-on-error=false --allow-gpu-metrics=true --log-verbose=1 \
       > /workspace/triton.log 2>&1 &
     ```
- Vérification depuis la VM : `curl -s localhost:8000/v2/health/ready` (OK), `tail -f /workspace/triton.log` pour suivre les requêtes. Pensez à fermer le job (`ovhai job stop a73355e1-70f0-421b-a25b-05a3d21ff266`) une fois les tests terminés.

## Procédure « tout-en-un » sur un job OVH
Objectif : construire `model.plan`, lancer Triton et récupérer le fichier final sans reconstruire d’image Docker custom. Tout est réalisé dans un seul job GPU.

1. **Démarrer le job GPU** (volume `standalone` monté en lecture/écriture, ports Triton exposés) :
   ```bash
   ovhai job run --name triton-inline \
     --gpu 1 --flavor a10-1-gpu \
     --unsecure-http --default-http-port 8000 \
     -v standalone:/workspace:rw \
     nvcr.io/nvidia/tritonserver:24.08-py3 \
     -- bash -lc "sleep infinity"
   ```
2. **Ouvrir un shell dans le conteneur** :
   ```bash
   ovhai job exec triton-inline -- bash
   ```
   Dans ce shell :
   ```bash
   set -euo pipefail
   apt-get update && apt-get install -y git
   cd /workspace
   rm -rf rerank-triton
   git clone https://github.com/M-kasinski/rerank-triton.git
   cd rerank-triton
   python3 -m pip install --no-cache-dir \
     torch==2.4.0 transformers==4.44.2 tokenizers==0.19.1 \
     huggingface_hub==0.24.6 onnx==1.16.1
   scripts/download_assets.sh
   scripts/build_trt_plan.sh
   ```
   `model.plan` est alors généré dans `/workspace/rerank-triton/models/mxbai_xs_trt/1/`.
3. **Lancer Triton directement depuis le job** (optionnel pour valider le moteur) :
   ```bash
   tritonserver \
     --model-repository=/workspace/rerank-triton/models \
     --http-port=8000 --grpc-port=8001 --metrics-port=8002 \
     --exit-on-error=false --log-verbose=1
   ```
   Tester ensuite les endpoints depuis la machine locale (`curl` sur `localhost:8000` via le port exposé par OVH).
4. **Rapatrier `model.plan`** (copie vers un bucket depuis le même job ou via un job CPU) :
   ```bash
   ovhai job run --name fetch-plan --gpu 0 \
     -v standalone:/workspace:ro \
     -v rerank-triton-artifacts@GRA:/mnt/output:rw \
     ubuntu:22.04 -- bash -lc \
     'cp /workspace/rerank-triton/models/mxbai_xs_trt/1/model.plan /mnt/output/model.plan'
   ```
   Puis, en local :
   ```bash
   ovhai bucket object download rerank-triton-artifacts@GRA model.plan ./model.plan
   ```
   (Adapter la commande si la CLI pose toujours problème : utiliser une autre machine ou monter le volume autrement.)

> Tentative locale le 31/10/2025 à 15h14 UTC-4 : `ovhai job run --name triton-inline ...` échoue immédiatement avec le même panic `system-configuration` (`Attempted to create a NULL object`). Impossible de valider la procédure tant que la CLI n’est pas exécutée depuis un environnement sain (VM Linux, conteneur, poste différent).

## Prochaines étapes suggérées
1. **Démarrer un Triton de validation sur OVH** : lancer un job basé sur `nvcr.io/nvidia/tritonserver:24.08-py3`, copier `model.plan` depuis le bucket `rerank-triton-artifacts@GRA`, exécuter `scripts/download_assets.sh`, puis démarrer `tritonserver` pour valider les endpoints via les payloads JSON fournis.
2. **Comparer les sorties** : rejouer les requêtes `payload_full.json` et `rerank_query_full.json`, vérifier la forme (`scores`, `indices`) et consigner les résultats dans le dépôt (ex. `docs.md` ou un rapport de test).
3. **Décider de la stratégie de stockage** : choisir entre garder `model.plan` dans le repo (Git LFS conseillé) ou s’appuyer uniquement sur le bucket OVH, puis ajuster README/guide opérateur.
4. **Réactiver/adapter la CI** : remettre en place le workflow Docker/Triton (build + push éventuel) et, si besoin, ajouter un job manuel pour déployer un serveur de test sur OVH.
5. **Nettoyer l’environnement OVH** : supprimer les jobs et volumes obsolètes, documenter la procédure `ovhai job run ... triton-test` dans le README, et archiver les commandes utiles pour les prochaines rotations.

## Commandes utilisées
```bash
# Inspection / modification locale
git status -sb
git diff
scripts/download_assets.sh
pip install onnx==1.16.1

# Désactivation CI
sed -n '1,80p' .github/workflows/docker-image.yml

# Jobs OVH pour la conversion TensorRT
ovhai job run --gpu 1 --flavor a10-1-gpu -v standalone:/workspace:rw \
  nvcr.io/nvidia/tensorrt:24.08-py3 -- bash -lc '<script>'
ovhai job logs <job-id>
ovhai job get <job-id>
ovhai volume list
ovhai datastore list
ovhai bucket create GRA rerank-triton-artifacts

# Job de copie vers l’Object Storage (échec)
ovhai job run --name copy-trt-plan --gpu 0 \
  -v standalone:/workspace:ro -v rerank-triton-artifacts@GRA:/mnt/output:rw \
  ubuntu:22.04 -- bash -lc 'cp /workspace/.../model.plan /mnt/output/model.plan'

# Jobs utilitaires pour inspection du volume standalone
ovhai job run --gpu 0 -v standalone:/workspace:ro ubuntu:22.04 -- bash -lc 'ls -R /workspace | head'
ovhai job run --gpu 0 -v standalone:/workspace:ro ubuntu:22.04 -- bash -lc 'find /workspace -maxdepth 5 -name model.plan -print'

# Jobs "build-trt-plan-upload" (en cours d'investigation)
ovhai job run --name build-trt-plan-upload --gpu 1 --flavor a10-1-gpu \
  nvcr.io/nvidia/tensorrt:24.08-py3 -- bash -lc '... trtexec ... sleep 3600'

# Liste/suivi des jobs
ovhai job list
# Crash CLI récent
RUST_BACKTRACE=1 ovhai job list
```
