---
slug: fauteuil-audio2led
domain: audio-led-visualizer
target_components: 64
min_components: 30
required_components:
  - lib_id_match: "Amplifier_Operational:LM358*"
    role: "audio frontend op-amp (3 stages)"
    min_count: 1
  - lib_id_match: "Connector:AudioJack3"
    role: "stereo line-level audio input (3.5 mm jack)"
    min_count: 1
  - lib_id_match: "Connector:Jack-DC"
    role: "DC barrel power input"
    min_count: 1
  - lib_id_match: "Regulator_Linear:L7805*"
    role: "5 V linear regulator"
    min_count: 1
  - lib_id_match: "IRF3415*"
    role: "power N-MOSFET LED driver"
    min_count: 1
  - lib_id_match: "Connector:Screw_Terminal_01x02"
    role: "LED channel screw terminal output"
    min_count: 4
  - lib_id_match: "Device:R_Potentiometer"
    role: "user-tunable gain / threshold pot"
    min_count: 2
schematic_format: "KiCad 6 (version 20211123)"
client_confidential: true
publication: "internal F4L benchmarking only"
---

# fauteuil-audio2led — Audio-to-LED Visualizer Driver

**Source** : `/Users/electron/Documents/Lelectron_rare/Fauteuil_Hypnotherapie/Audio2LED PCB/Audio2LED PCB.kicad_sch`
**Schematic format** : KiCad 6 (`(version 20211123)`)
**Component count** : 64 symbols (per `(in_bom yes)`), 12 distinct
`lib_id` classes
**Project status** : legacy 2022 driver board — superseded in production
by the DALI PCB (`hypnoled_PCB/hypnoled/DALI PCB/`) but still a clean,
self-contained reference design for the audio frontend + LED MOSFET
chain.

## Description

The `Audio2LED PCB` is a **single-sheet analog audio-to-LED visualizer
driver** designed for the *Fauteuil Hypnothérapie* (hypnotherapy chair)
prototype. It takes a stereo line-level audio signal in, conditions it
through three LM358 op-amp stages, and drives high-current LED strips
through power N-MOSFETs. There is no MCU on this board : the LED
brightness response is purely analog (envelope-follower style), tuned by
front-panel potentiometers.

Architectural role :

```
3.5 mm stereo jack ──► AC coupling ──► LM358 stage A (gain)
                                                │
                                                ├──► LM358 stage B (rectifier / envelope)
                                                │              │
                                                │              ▼
                                                │     LM358 stage C (drive)
                                                │              │
                                                │              ▼
                                                │   IRF3415 N-MOSFET ──► LED chain @ +12 V
                                                │
                                                └──► trim pots (gain + threshold)

Jack-DC ──► L7805 ──► +5 V rail (op-amp Vcc, midrail bias)
        └──► +12 V rail (LED + MOSFET drain)
```

Six screw-terminal pairs expose the LED channels (common +12 V plus
switched return), allowing the chair integrator to wire arbitrary LED
loads up to the MOSFET current limit.

## Inputs

| # | Signal | Range | Sensing element |
|---|--------|-------|-----------------|
| 1 | Stereo line-level audio | ~1 Vrms typical (consumer line out) on 3.5 mm jack tip + ring | Connector:AudioJack3, AC-coupled via C_Polarized |
| 2 | DC power | 12 V nominal (estimate ; **TBD** confirm against jack rating + L7805 input headroom) | Connector:Jack-DC |
| 3 | User gain trim | 4 × `R_Potentiometer` (front-panel, ~10 kΩ estimate ; **TBD** confirm value) | analog control of LM358 feedback / bias |

## Outputs

| # | Signal | Description |
|---|--------|-------------|
| 1 | Switched-low LED returns × 6 | Six `Screw_Terminal_01x02` blocks providing common +12 V rail + per-channel MOSFET-switched ground (drives external LED strips) |
| 2 | LED drive current per channel | bounded by the IRF3415 (continuous Id ≈ 27 A in TO-220, but in this PCB realistically limited by trace + pad thermal budget — **TBD** measure on bench) |
| 3 | Visual envelope mapping | LED brightness ∝ rectified audio envelope after gain + threshold (**not** FFT — this is a pure analog envelope follower, no spectral split) |

## Constraints

| # | Constraint | Target / value |
|---|-----------|----------------|
| C1 | BOM cost (prototype, qty = 5, JLCPCB) | ≤ 30 € per board (estimate ; **TBD** — pull exact figure from `Audio2LED PCB/jlcpcb/` if BOM CSV present) |
| C2 | PCB outline | ~80 mm × 60 mm (estimate ; **TBD** verify against `Audio2LED PCB.kicad_pcb` board edge) |
| C3 | Component count | 64 symbols, 12 distinct `lib_id` classes (matches CATALOG slot 5) |
| C4 | MCU family | **none** — fully analog signal chain (no microcontroller on this board) |
| C5 | Audio bandwidth | ~20 Hz - 20 kHz nominal (LM358 GBW = 1 MHz, ample for envelope detection) |
| C6 | Power budget | LED strip load dominates (~12 V × N × I_per_channel) ; control side (3 × LM358 + bias) is < 50 mA from 5 V rail |
| C7 | Operating temperature | 0 °C to +50 °C ambient (indoor wellness room conditions) |
| C8 | Mechanical mounting | enclosed in `hypnoled hammon 1591B` Hammond box (per project STL) |

## Acceptance criteria

| # | Check | Tool / threshold |
|---|-------|------------------|
| A1 | Schematic ERC clean | `kicad-cli sch erc gold.kicad_sch` → zero `severity == "error"` items (warnings allowed but tracked) |
| A2 | Audio frontend topology | exactly 3 × LM358 stages cascaded between the 3.5 mm jack input and the MOSFET gates (gain → rectify → drive). Missing or extra stages = fail. |
| A3 | LED drive topology | each output channel routes through one IRF3415 N-MOSFET in low-side switching configuration (drain to LED return, source to GND, gate driven by the LM358 envelope output) |
| A4 | Power conditioning | one L7805 linear regulator from +12 V → +5 V for the op-amp supply ; +12 V passed straight through to LED rail |
| A5 | Output count | 6 × `Screw_Terminal_01x02` providing LED channel breakouts ; mismatched count = fail |
| A6 | AC coupling on audio path | series `C_Polarized_Small` between jack input and first LM358 stage (DC-blocking) |
| A7 | User trim | at least 2 (ideally 4) `R_Potentiometer` instances on the LM358 feedback / bias network |

A1 + A2 + A3 + A5 are the **mandatory** gates for considering an
LLM-emitted candidate "spec-compliant" ; A4, A6, A7 are graded soft
criteria for the scoring rubric.

## Compliance

- **Grade** : prototype / pre-production hardware. No CE / FCC / UL
  certification required at this stage. An expert report
  (`Rapport d'Expertise*.pdf`) was commissioned in 2023 to assess
  repairability + safety — see project root for findings.
- **RoHS** : target compliance for all sourced parts (JLCPCB basic
  catalog parts are RoHS by default ; LM358 / L7805 / IRF3415 are
  long-standing RoHS parts).
- **Safety classification** : SELV (12 V supply), no functional safety
  standard claimed.
- **Audio safety** : line-level input only ; no speaker / headphone
  amplification.
- **Open-source status** : **client-confidential** project. Internal
  F4L benchmarking is fine ; **do not** publish the schematic in
  external papers / public repos without owner approval.

## TBD / open questions

These fields were marked `TBD` above and need a follow-up pass before
this slot is used as a strict gold reference :

1. **C1** — exact BOM cost (read `Audio2LED PCB/jlcpcb/bom.csv` if
   present)
2. **C2** — exact PCB outline dimensions (read
   `Audio2LED PCB.kicad_pcb` board edge)
3. Input #2 — exact DC jack input voltage rating
4. Input #3 — exact potentiometer resistance value
5. Output #2 — bench-measured per-channel current limit (PCB trace
   thermal budget vs IRF3415 rating)
6. Whether to copy `Audio2LED PCB.kicad_pcb` into the corpus slot
   (currently schematic-only)
7. Exact output channel count (6 from screw-terminal count, but cross-
   check against MOSFET count = 2 — there may be channel grouping or
   parallel terminals per MOSFET ; **TBD** clarify by reading the
   `kicad_pcb`)
8. Stage allocation : confirm whether the 3 LM358 packages are wired
   as 3 dual-stages (6 op-amps total) or whether some channels share
   stages (impacts A2 strict-compliance check)

## Provenance

- Hardware design : L'Electron Rare (Clément Saillant)
- Client : private wellness practitioner (Fauteuil Hypnothérapie
  prototype, Garnier client)
- Project tree : `~/Documents/Lelectron_rare/Fauteuil_Hypnotherapie/`
- Status : superseded by the DALI PCB design
  (`hypnoled_PCB/hypnoled/DALI PCB/`) for production ; kept here as a
  clean analog reference
- License : project-internal — **not** open-sourced
- Open-source status : **client-confidential** (per CATALOG.md
  "Risques" section). Slot is restricted to **internal F4L
  benchmarking** only.
