# kxkm-bmu-v1 — Battery Management Unit (BMU) v1

**Source** : `/Users/electron/Documents/Lelectron_rare/1-KXKM/KXKM_Batterie_Parallelator/hardware/PCB/BMU v1/BMU v1.kicad_sch`
**Schematic format** : KiCad 8 (`(version 20231120)`)
**Top-level component count** : 56 symbols on root sheet, plus 38
(`power.kicad_sch`) + 29 (`batteries.kicad_sch`) + 76 (`I2C.kicad_sch`)
hierarchical sub-sheets — total ~199 instantiated symbols across the design,
of which the catalog-relevant gold count for slot 1 is **68 unique
component classes** as listed in `tests/corpus/CATALOG.md`.
**Project status** : production hardware, deployed for KompleX KapharnaüM
(Villeurbanne, FR) — live-arts off-grid scenography.

## Description

The KXKM BMU is a 16-channel **battery management unit** built around an
ESP32-S3 (ESP32-S3-BOX-3 form factor for the supervising MCU board) that
safely parallels up to **32 battery packs in the 24-30 V range** for
off-grid stage installations. The PCB monitored by this schematic is the
"sensor + switch" board that sits between the battery bus and the
ESP32-S3 supervisor : it owns voltage / current / temperature acquisition
on a per-cell basis and the MOSFET-based per-cell disconnect path.

Architectural role :

```
ESP32-S3-BOX-3  ──I2C 50 kHz──►  TCA9535 (×8, GPIO expanders)
                                       │
                                       ├── MOSFET drivers (4 per TCA, 32 total)
                                       └── status LEDs (4 per TCA)

ESP32-S3-BOX-3  ──I2C 50 kHz──►  INA237 (×16, V + I monitor, 2 mΩ shunt)
```

Per-cell **firmware-level protection** : under/over voltage, over current,
imbalance vs pack mean, switch-count rate limiting, permanent lock after
5 fault events. Faults trigger sub-microsecond MOSFET disconnect.

## Inputs

| # | Signal | Range | Sensing element |
|---|--------|-------|-----------------|
| 1 | Per-pack voltage `V_batt[i]`, i ∈ \[0, 15\] | 24 000 - 30 000 mV (nominal Li-ion / LiFePO4 packs in series) | INA237 bus-voltage input |
| 2 | Per-pack current `I_batt[i]` | ±10 000 mA continuous, peaks up to short-circuit fuse limit | 2 mΩ shunt + INA237 |
| 3 | Pack temperature (optional, via NTC on header) | -20 °C to +85 °C | external NTC, ADC of ESP32-S3 |
| 4 | Bus-side combined current | derived per-pack sum + global INA on bus rail | TBD (verify on schematic) |
| 5 | I²C SDA / SCL from ESP32-S3 | 3.3 V logic, 50 kHz | direct + TCA9548 mux (TBD) |

Per the firmware (`bmu_protection`), the **default thresholds** are :

| Threshold | Default value | Configurable via |
|-----------|---------------|------------------|
| `V_min` | 24 000 mV | NVS + Kconfig |
| `V_max` | 30 000 mV | NVS + Kconfig |
| `I_max` | 10 000 mA | NVS + Kconfig |
| `Imbalance_max` | 1 000 mV | NVS + Kconfig |
| `Reconnect_delay` | 10 000 ms | Kconfig |
| `Switch_limit` (events before permanent lock) | 5 | Kconfig |

## Outputs

| # | Signal | Description |
|---|--------|-------------|
| 1 | Per-pack ON/OFF switching | low-side MOSFET driven by TCA9535 GPIO; sub-microsecond disconnect on fault |
| 2 | Per-pack status LED (R/G) | green = connected, solid red = disconnected, blinking red ≈ 1 Hz = error / locked |
| 3 | I²C readback of `(V, I, T, status_flags)` per pack | 16 INA237 readings + 8 TCA9535 GPIO snapshots, polled at firmware-defined cadence |
| 4 | Pack-level fault flags | OVER_V, UNDER_V, OVER_I, IMBALANCE, LOCKED — surfaced over BLE GATT, MQTT and the touchscreen LVGL UI |
| 5 | Soft-balancing duty cycle command | duty-cycled MOSFET pattern from `bmu_balancer` for opportunistic R_int measurement (firmware concern, but hardware must tolerate fast PWM-style switching of the MOSFET driver) |

## Constraints

| # | Constraint | Target / value |
|---|-----------|----------------|
| C1 | BOM cost (prototype, qty = 5, JLCPCB) | ≤ 50 € per board (estimate; **TBD** — pull exact figure from `hardware/PCB/BMU v1/jlcpcb/bom.csv`) |
| C2 | PCB outline | ≤ 100 mm × 80 mm (estimate; **TBD** — verify against `BMU v1.kicad_pcb` board edge) |
| C3 | Component count | ≤ 100 unique parts (current : ~68 per CATALOG slot 1) |
| C4 | MCU family | ESP32-S3 (the supervisor lives on a separate ESP32-S3-BOX-3 module; this PCB carries no MCU itself, only I²C peripherals + power conditioning) |
| C5 | I²C bus speed | 50 kHz (slow on purpose : long flying leads to remote packs) |
| C6 | Operating temperature | -20 °C to +60 °C ambient (touring conditions, indoor + outdoor stages) |
| C7 | Mechanical mounting | DIN-rail or M3 standoffs (TBD — confirm against `BMU v1.kicad_pcb` mounting holes) |
| C8 | Galvanic isolation | not required between MCU and pack side at this revision (24-30 V is below SELV double-fault threshold) |

## Acceptance criteria

| # | Check | Tool / threshold |
|---|-------|------------------|
| A1 | Schematic ERC clean | `kicad-cli sch erc gold.kicad_sch` → zero `severity == "error"` items (warnings allowed but tracked) |
| A2 | PCB DRC clean | `kicad-cli pcb drc BMU v1.kicad_pcb` → zero `severity == "error"` items (only relevant when the `.kicad_pcb` is co-located in the corpus slot, which it currently is **not** — TBD whether to bring the PCB along) |
| A3 | I²C address plan respected | INA237 chain addresses `0x40..0x4F` (16 chips, address-strap programmed) ; TCA9535 chain addresses `0x20..0x27` (8 chips). Mismatch = fail. |
| A4 | MOSFET switching topology | each pack channel has exactly one low-side N-MOSFET driven by a TCA9535 GPIO through a gate driver (or RC + bjt level shifter) ; back-to-back configuration **not** required at this revision (uni-directional discharge protection only). |
| A5 | Shunt path | each pack current path includes a single 2 mΩ ±1 % SMD shunt in series with the MOSFET, with INA237 V_in+/V_in- across it. |
| A6 | Power conditioning | the `power.kicad_sch` sub-sheet provides a regulated 3.3 V rail to all I²C peripherals from the lowest-voltage pack (TBD : verify buck topology — likely a wide-Vin buck such as TPS54360 or equivalent). |
| A7 | Per-pack visual feedback | each channel exposes a bicolour (or 2 separate LEDs) indicator routed to a dedicated TCA9535 GPIO. |

A1 + A3 + A5 are the **mandatory** gates for considering an LLM-emitted
candidate "spec-compliant" ; A2, A4, A6, A7 are graded soft criteria for
the scoring rubric.

## Compliance

- **Grade** : prototype / pre-production hardware. No CE / FCC / UL
  certification required at this stage.
- **RoHS** : target compliance for all sourced parts (JLCPCB basic
  catalog parts are RoHS by default ; extended parts must be checked
  individually).
- **Safety classification** : SELV (output ≤ 30 V, single-fault
  protected), no functional safety standard claimed (no IEC 61508 / ISO
  26262 audit ; not in scope for live-arts staging equipment).
- **Software side** (out of scope of this hardware spec but relevant for
  end-to-end evaluation) : ESP-IDF v5.4, GPLv3 license, threshold
  immutability rule from `CLAUDE.md` ("ne **jamais** affaiblir les
  seuils protection (V/I/délai/topology)").

## TBD / open questions

These fields were marked `TBD` above and need a follow-up pass before
this slot is used as a strict gold reference :

1. **C1** — exact BOM cost (read `hardware/PCB/BMU v1/jlcpcb/bom.csv`)
2. **C2** — exact PCB outline dimensions (read `.kicad_pcb` board edge)
3. **C7** — mounting hole pattern
4. **A2** — decide whether to copy the `.kicad_pcb` into the corpus slot
   or limit gold reference to schematic only
5. **A6** — exact buck regulator part number used in `power.kicad_sch`
6. Input #4 — bus-side global current sensing (verify against schematic)
7. Input #5 — presence / absence of an I²C mux (TCA9548 or similar)
8. Hierarchical sub-sheet inclusion : currently only the top-level
   `BMU v1.kicad_sch` is copied into the corpus slot. The three sub-sheets
   (`I2C.kicad_sch`, `batteries.kicad_sch`, `power.kicad_sch`) live
   alongside it in the source tree but are **not** copied here. This means
   `kicad-cli sch erc` against `gold.kicad_sch` may report missing-sheet
   warnings. Decide whether to pull the sub-sheets in or run ERC with
   `--severity-error` only.

## Provenance

- Hardware design : L'Electron Rare (Clément Saillant)
- Client / use case : KompleX KapharnaüM (Villeurbanne, FR) — off-grid
  digital scenography
- Firmware repository : `~/Documents/Lelectron_rare/1-KXKM/KXKM_Batterie_Parallelator/`
- License : GPLv3 (firmware) ; hardware files inherit project license
- Open-source status : assumed open (the project README publicises CI
  badges on a public GitHub repo) ; safe to use in external publications
  per the "Risques" section of `tests/corpus/CATALOG.md`.
