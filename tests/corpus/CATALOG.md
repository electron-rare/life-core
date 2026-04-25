# F4L AI EDA Evaluation Corpus — CATALOG

**Date** : 2026-04-25 (mis à jour 2026-04-25 15:35)
**Statut** : 3/10 slots bootstrappés (slot 1 `kxkm-bmu-v1`, slot 2
`fauteuil-audio2led`, slot 3 `amis-i2c-spi-current-ctrl`). 7 restent à
écrire pour le gate ADR-007.
**Références** :
- **ADR-007** (F4L parent) — gate sur ce corpus (10 deliverables ≥ 30 composants)
- **ADR-009** (F4L parent) — convention YAML frontmatter pour `spec.md`. Slots 2-3 conformes ; slot 1 retrofit pending (follow-up #2 d'ADR-009).
- **Baseline empirique** (F4L parent `reports/baseline-2026-04-25-3way-3slots.md`) — campagne 6-way × 3 slots × n=3 du 2026-04-25 : Devstral fine-tuned **88.9 %**, Mistral Small 77.8 %, Claude Sonnet 100 %, Qwen 35B raw 66.7 %. Métric actuel sature à Claude — `required_components_coverage` (PR #18, exige frontmatter) discriminera mieux dès slot 1 retrofit.

## But

ADR-007 conditionne la décision d'investir dans un POC LangGraph derrière une
mesure baseline du `kicad_generator` actuel sur **10 deliverables hardware ≥ 30
composants** chacun. Sans corpus, le gate (« si gain cumulé P1+P2+P3 ≥ +10 pp
DRC-clean rate, freeze patches et skip LangGraph ; sinon proceed POC ») est
inexécutable.

Ce catalogue énumère les candidats déjà disponibles sur le filesystem local et
signale les actions requises pour rendre chaque entrée corpus-ready.

## Méthodologie d'exécution (couplée à `scripts/measure_drc_clean_rate.py`)

Pour chaque entrée :

1. La spec (Markdown ou PRD) est lue depuis `<corpus_dir>/<slot>/spec.md`
2. Le gold schematic (KiCad sch) est posé à `<corpus_dir>/<slot>/gold.kicad_sch`
3. Optionnellement, un `<slot>/gold.json` (forme attendue de l'émission LLM)
4. Le script `measure_drc_clean_rate.py` itère sur chaque slot, lance
   `KicadGenerator.generate(GenerationContext(slug=<slot>, ...))` `n_runs`
   fois, capture Pass@1 / Pass@k / latence / tokens / coût / classes d'erreur

## Inventaire des 11 candidats ≥ 30 composants

| # | Nom slot | Source filesystem | Composants | Domaine | Spec | Gold sch | Effort |
|---|----------|-------------------|-----------:|---------|------|----------|-------:|
| 1 | `kxkm-bmu-v1` | `/Users/electron/Documents/Lelectron_rare/1-KXKM/KXKM_Batterie_Parallelator/hardware/PCB/BMU v1/BMU v1.kicad_sch` | 68 | BMS power | TO WRITE (réutiliser `KXKM_Parallelator/README.md`) | exists, copier | 2 j-h |
| 2 | `kxkm-bmu-v2` | `/Users/electron/Documents/Lelectron_rare/1-KXKM/KXKM_Batterie_Parallelator/hardware/pcb-bmu-v2/BMU v2.kicad_sch` | 66 | BMS power v2 | TO WRITE (delta vs v1) | exists | 2 j-h |
| 3 | `kxkm-i2c-repeater` | `/Users/electron/Documents/Lelectron_rare/1-KXKM/KXKM_Batterie_Parallelator/hardware/pcb-bmu-v2/I2C_repeter/I2C_repeter.kicad_sch` | 81 | Interface I2C bus extension | TO WRITE | exists | 2 j-h |
| 4 | `fauteuil-hypnoled` | `/Users/electron/Documents/Lelectron_rare/Fauteuil_Hypnotherapie/PCB/hypnoled.kicad_sch` | 64 | LED control hypnothérapie | TO WRITE | exists | 2 j-h |
| 5 | `fauteuil-audio2led` | `/Users/electron/Documents/Lelectron_rare/Fauteuil_Hypnotherapie/Audio2LED PCB/Audio2LED PCB.kicad_sch` | 64 | Audio→LED visualizer | **bootstrapped 2026-04-25** | bootstrapped | done |
| 6 | `ledcurtain-teensy` | `/Users/electron/Documents/Lelectron_rare/LEDcurtain_hardware/Teensy Board/Teensy Board.kicad_sch` | 117 | MCU control board (Teensy) | TO WRITE | exists | 3 j-h |
| 7 | `ledcurtain-led-driver` | `/Users/electron/Documents/Lelectron_rare/LEDcurtain_hardware/LED Board/LED Board.kicad_sch` | 99 | LED matrix driver | TO WRITE | exists | 2 j-h |
| 8 | `amis-i2c-spi-current-ctrl` | `/Users/electron/Documents/Lelectron_rare/Les_Amis_Nos_Morts/amis nos morts PCB/I2C_SPI_current-control.kicad_sch` | 100 | Mixed-signal current control | **bootstrapped 2026-04-25** | bootstrapped | done |
| 9 | `amis-io-board` | `/Users/electron/Documents/Lelectron_rare/Les_Amis_Nos_Morts/amis nos morts PCB/IO.kicad_sch` | 119 | I/O expansion board | TO WRITE | exists | 3 j-h |
| 10 | `amis-i2c-pot` | `/Users/electron/Documents/Lelectron_rare/Les_Amis_Nos_Morts/amis nos morts PCB/I2C_POT.kicad_sch` | 95 | I2C potentiometer | TO WRITE | exists | 2 j-h |
| 11 | `super-mixer-eq` | `/Users/electron/Documents/Projets_Creatifs/L_Electron_Fou/04_MATIERES_PREMIERES/creation-electronique/Super mixer/eq.kicad_sch` | 43 | Audio EQ analog | TO WRITE | exists | 2 j-h |

**Total slots disponibles** : 11 (1 de réserve)
**Total composants cumulés** : ~1 016 across all slots
**Range** : 43 → 119 composants par slot

## Distribution thématique

| Domaine | Slots | Diversité |
|---|---:|---|
| BMS / power | 3 (slots 1-3) | KXKM stack, FR original |
| LED control | 3 (slots 4-7 partiel) | Fauteuil + LEDcurtain, FR original |
| Mixed-signal / capteurs | 2 (slots 8, 10) | Les_Amis_Nos_Morts |
| MCU board | 1 (slot 6) | Teensy |
| I/O | 1 (slot 9) | Les_Amis_Nos_Morts |
| Audio analog | 1 (slot 11) | Super mixer |

**Verdict diversité** : acceptable mais pas idéale — heavy on KXKM + Les_Amis_Nos_Morts. À long terme, ajouter sensor-nodes (ESP32, RP2040, STM32), motor drivers, USB power delivery, RF (BLE/LoRa), pour couvrir les classes manquantes.

## Statut résumé

- ✅ **11 / 10 slots avec gold schematic disponible** sur disque
- 🟡 **3 / 10 specs Markdown PRD écrites** — slots `kxkm-bmu-v1`,
  `fauteuil-audio2led`, `amis-i2c-spi-current-ctrl` (effort moyen
  ~2 j-h par slot, total ~25 j-h pour tout le corpus)
- 🟡 **3 / 10 entries posées dans `life-core/tests/corpus/<slot>/`**
  (slot 1 + slot 5 + slot 8) — 7 slots restants à bootstrapper

### Phase 2 progress (Phase 2 priority order)

| Step | Slot | Status |
|------|------|--------|
| Phase 1 | `kxkm-bmu-v1` | ✅ bootstrapped (PR #15) |
| Phase 2.1 | `fauteuil-audio2led` | ✅ bootstrapped 2026-04-25 |
| Phase 2.2 | `amis-i2c-spi-current-ctrl` | ✅ bootstrapped 2026-04-25 |
| Phase 2.3 | `kxkm-i2c-repeater` | ⏳ next |
| Phase 2.4 | `ledcurtain-teensy` | ⏳ pending |
| Phase 2.5 | `super-mixer-eq` | ⏳ pending |
| Phase 2.6 | `kxkm-bmu-v2` | ⏳ pending |
| Phase 2.7 | `amis-io-board` | ⏳ pending |
| Phase 2.8 | `fauteuil-hypnoled` | ⏳ pending |
| Phase 2.9 | `ledcurtain-led-driver` | ⏳ pending |

## Action requise pour rendre le corpus opérationnel

### Phase 1 — Bootstrapping minimal (1 j-h, débloque la mesure)

Pour pouvoir lancer `measure_drc_clean_rate.py` immédiatement avec **1 deliverable réel** (au-delà de sensor-node-minimal qui est sous le seuil) :

1. Choisir slot 1 `kxkm-bmu-v1` (le plus mature, bien documenté, IP utilisateur)
2. `mkdir life-core/tests/corpus/kxkm-bmu-v1/`
3. Copier `BMU v1.kicad_sch` → `life-core/tests/corpus/kxkm-bmu-v1/gold.kicad_sch`
4. Écrire un `spec.md` minimal (1-2 pages) à partir du `KXKM_Parallelator/README.md`
5. Lancer `measure_drc_clean_rate.py --corpus-dir life-core/tests/corpus/ --n-runs 5` pour première baseline réelle

### Phase 2 — Compléter à 10 slots (~25 j-h, débloque le gate ADR-007)

Travailler les 9 slots suivants un par un (priorité diversité) :
1. `fauteuil-audio2led` (audio→visual, contraste avec BMS)
2. `amis-i2c-spi-current-ctrl` (mixed-signal SPI/I2C)
3. `kxkm-i2c-repeater` (interface bus)
4. `ledcurtain-teensy` (MCU substantiel, ~117 composants)
5. `super-mixer-eq` (audio analog pur)
6. `kxkm-bmu-v2` (évolution v1 → variation incrémentale utile)
7. `amis-io-board` (I/O massive, ~119 composants)
8. `fauteuil-hypnoled` (LED control simple)
9. `ledcurtain-led-driver` (LED matrix driver)

Pour chaque slot : `mkdir`, copier sch, écrire spec ~1-2 pages depuis README/notes existants. Délégable à un agent mais nécessite des allers-retours sur la spec.

### Phase 3 — Synthétiques pour combler les gaps thématiques (long-terme)

Ajouter au moins :
- 1 sensor-node ≥ 30 composants (étendre `sensor-node-minimal` actuel)
- 1 USB Power Delivery ≥ 30 composants
- 1 RF module (BLE/LoRa/WiFi)
- 1 motor driver (BLDC ou stepper)

Cela porterait le corpus à 14-15 slots, dépassant largement le minimum 10 d'ADR-007.

## Recommandation immédiate

**Commencer par slot 1 `kxkm-bmu-v1`** — c'est le candidat le plus mature, le mieux documenté, et le plus aligné avec le récit stratégique « F4L est l'outil naturel pour les BMS ». Une fois la première mesure baseline obtenue (~1 j-h), continuer slot par slot en parallèle des autres travaux.

## Risques de cette approche

- **Pas tous open-source** : Les_Amis_Nos_Morts et Fauteuil_Hypnotherapie sont des projets clients confidentiels. Ne **pas** publier les schémas dans une publi ou benchmark public sans accord. Pour publi externe, n'utiliser que KXKM (open-source assumé) + sensor-node-minimal.
- **Hétérogénéité KiCad version** : certains schémas sont KiCad 7 ou 6 (à vérifier au cas par cas via header `(version ...)`). Le `kicad_generator` cible KiCad 8. Convertir au besoin via `kicad-cli sch upgrade` (manuelle).
- **Specs PRD à inventer** : aucun de ces 11 projets n'a une spec Markdown formelle. Reverse-engineering depuis README + commentaires schéma + mémoire utilisateur. Compter ~2 j-h par spec.
- **Drift gold schematic vs spec** : si le LLM est strictement noté contre le gold, il sera pénalisé pour des choix légitimes différents. Évaluer aussi via DRC-clean rate (pas juste similarité au gold).
