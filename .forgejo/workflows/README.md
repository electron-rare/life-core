# life-core -- Forgejo Actions

Runs on `docker` self-hosted label (live runner
`f4l-forgejo-runner-01`, image `gitea/act_runner`, deployed at
`/home/electron/forgejo-runner/` on electron-server). The hardened
source-of-truth compose lives at `ops/forgejo-runner/` in the F4L
monorepo and will replace the live runner in V1.9; that runner
exposes `f4l-python` instead of `docker`.

Secrets referenced via `${{ secrets.* }}` are pre-populated from
Infisical by `scripts/forgejo-actions-seed-secrets.sh` (Task 10).

GitHub Actions remains authoritative for public repos; this file
enables CI on `git.saillant.cc` for the F4L private source of truth
established in V1.7.
