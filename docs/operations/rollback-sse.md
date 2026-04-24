# Rollback SSE /events

Si `/events` cause des problèmes (Traefik timeout, crash life-core, etc.) :

## Option A (env var, requires compose edit + restart)

1. Éditer `docker-compose.prod.yml`, section `life-core.environment:` :

   ```yaml
   - F4L_UI_FEATURE_SSE=true   # ← changer en "false"
   ```

2. Appliquer :

   ```bash
   ssh -J kxkm@kxkm-ai electron-server \
     "cd ~/lelectron-rare/factory-4-life && \
      docker compose -f docker-compose.prod.yml up -d life-core"
   ```

3. Frontend détecte `ui_features.sse=false` via `/config/platform` au prochain
   fetch (< 60 s staleTime) et repasse en polling.

## Option B (override compose, sans édit du fichier principal)

1. Créer `docker-compose.override-rollback-sse.yml` :

   ```yaml
   services:
     life-core:
       environment:
         - F4L_UI_FEATURE_SSE=false
   ```

2. Restart avec override :

   ```bash
   ssh -J kxkm@kxkm-ai electron-server \
     "cd ~/lelectron-rare/factory-4-life && \
      docker compose -f docker-compose.prod.yml \
                     -f docker-compose.override-rollback-sse.yml \
                     up -d life-core"
   ```

## Option C (code revert)

```bash
git revert <commit_SHA_de_feat(events)_add_SSE_snapshot_stream>
git push origin feat/ui-consolidation-2026-04-23
# puis redeploy standard
```

## Validation après rollback

- Le Dashboard doit reprendre ses polls `/health` et `/stats` (visible dans
  DevTools Network)
- Aucune erreur console `EventSource`
