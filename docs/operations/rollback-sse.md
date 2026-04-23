# Rollback SSE /events

Si `/events` cause des problèmes (Traefik timeout, crash life-core, etc.) :

## Option A (rapide, sans redeploy)

```bash
ssh -J kxkm@kxkm-ai electron-server \
  "cd ~/lelectron-rare/factory-4-life && \
   docker compose -f docker-compose.prod.yml exec life-core \
   sh -c 'export F4L_UI_FEATURE_SSE=false'"
# Puis forcer restart de life-core pour que l'env prenne effet :
ssh -J kxkm@kxkm-ai electron-server \
  "docker compose -f ~/lelectron-rare/factory-4-life/docker-compose.prod.yml restart life-core"
```

Le frontend détecte le flag à `false` au prochain `/config/platform` (< 1 min)
et repasse en mode polling.

## Option B (code revert)

```bash
git revert <commit SHA du "feat(events): ...">
git push origin feat/kiki-router-integration
# puis redeploy standard
```

## Validation après rollback

- Le Dashboard doit reprendre ses polls `/health` et `/stats` (visible dans
  DevTools Network)
- Aucune erreur console `EventSource`
