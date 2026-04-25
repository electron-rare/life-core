---
slug: amis-i2c-spi-current-ctrl
domain: mixed-signal-current-control
target_components: 100
min_components: 60
required_components:
  - lib_id_match: "Interface_Expansion:PCA9555*"
    role: "I2C 16-bit GPIO expander (SPI chip-select decode)"
    min_count: 1
  - lib_id_match: "Potentiometer_Digital:MCP42100"
    role: "dual-channel SPI digital potentiometer (current setpoint)"
    min_count: 8
  - lib_id_match: "Power_Management:TJ2242"
    role: "constant-current LED driver (16 channels)"
    min_count: 16
  - lib_id_match: "Device:R_Small"
    role: "I2C pull-up / passive support"
    min_count: 1
schematic_format: "KiCad 6 (version 20211123)"
sheet_kind: "hierarchical sub-sheet"
parent_sheet: "amis nos morts PCB.kicad_sch"
client_confidential: true
publication: "internal F4L benchmarking only"
---

# amis-i2c-spi-current-ctrl — Mixed-Signal Current Control (16ch)

**Source** : `/Users/electron/Documents/Lelectron_rare/Les_Amis_Nos_Morts/amis nos morts PCB/I2C_SPI_current-control.kicad_sch`
**Schematic format** : KiCad 6 (`(version 20211123)`)
**Component count** : 100 symbols (per `(in_bom yes)`), 6 distinct
`lib_id` classes (heavy on power + digi-pot + LED-driver instances)
**Sheet kind** : hierarchical sub-sheet of `amis nos morts PCB.kicad_sch`
(communicates with the parent through 23 `hierarchical_label` ports —
`SCL_2`, `SDA_2`, `SI`, `SO`, `SCK`, `CS`, `INT`, `VO_1..VO_16`).
**Project status** : custom show electronics for the *Les Amis Nos
Morts* artistic project (Guillaume Dalin / lesamisnosmorts.fr,
v0.1a — June 2022).

## Description

The `I2C_SPI_current-control` board is the **mixed-signal current-
control core** of the *Les Amis Nos Morts* show controller. It exposes
**16 independently-controllable constant-current outputs** (`VO_1` ..
`VO_16`) intended to drive servomotor loads (per the OSC-to-I2C +
Adafruit-PWM-Servo-Driver pipeline documented at the parent project
level), with the per-channel current setpoint commanded over **SPI** via
a chain of digital potentiometers and the per-channel enable / sense
flow handled over **I²C** through a GPIO expander.

Architectural role :

```
parent sheet ──I²C (SDA_2 / SCL_2)──► PCA9555 GPIO expander
                                              │
                                              ├─ chip-select decode → CS lines
                                              └─ INT line back to parent

parent sheet ──SPI (SCK / SI / SO / CS)──► MCP42100 ×8 (16 wiper channels total)
                                              │
                                              ▼
                                      TJ2242 ×16 (constant-current
                                      LED / load driver, 1 per output)
                                              │
                                              ▼
                                       VO_1 .. VO_16 ──► hierarchical
                                       output back to parent sheet
```

The **8 × MCP42100** (each = 2 independent 100 kΩ digital wipers,
SPI-controlled) provide 16 analog setpoints feeding the 16 × TJ2242
constant-current sinks ; the PCA9555 GPIO expander handles chip-select
decoding + per-channel auxiliary logic + interrupt-back to the
supervisor.

## Inputs

| # | Signal | Type | Notes |
|---|--------|------|-------|
| 1 | I²C bus (`SDA_2`, `SCL_2`) | digital, ~3.3 V or 5 V (TBD) | hierarchical input from parent ; addresses PCA9555 (default `0x20` family) |
| 2 | SPI bus (`SCK`, `SI`, `SO`) | digital, ~5 V (TBD verify against MCP42100 Vdd) | shared SPI bus reaching all 8 × MCP42100 |
| 3 | SPI chip-select lanes (`CS`) | digital | derived per chip via PCA9555 GPIO decode (one CS per MCP42100) |
| 4 | Interrupt line (`INT`) | digital, open-drain | PCA9555 `INT` back to parent supervisor |
| 5 | +5 V supply | analog/digital rail | hierarchical input (powers MCP42100, PCA9555 logic, TJ2242 control side) |
| 6 | GND reference | reference | shared with parent |

## Outputs

| # | Signal | Description |
|---|--------|-------------|
| 1 | `VO_1` .. `VO_16` | 16 constant-current outputs (TJ2242 sinks), per-channel current set by the corresponding MCP42100 wiper voltage |
| 2 | Per-channel current range | TJ2242 nominal (TBD verify exact range from datasheet — typical 5 mA to ~700 mA per channel depending on Rext + wiper voltage) |
| 3 | Bus-side ack / status | PCA9555 readback over I²C (per-channel sense if wired ; **TBD** confirm whether sense is implemented) |

## Constraints

| # | Constraint | Target / value |
|---|-----------|----------------|
| C1 | BOM cost (prototype, qty = 5, JLCPCB) | ≤ 80 € per board (estimate ; **TBD** — pull exact figure from `amis nos morts PCB/jlcpcb/` if BOM CSV present) |
| C2 | PCB outline | shared with parent assembly (this is a hierarchical sub-sheet, not its own PCB outline) — not directly applicable |
| C3 | Component count | 100 symbols (matches CATALOG slot 8) |
| C4 | MCU family | **none on this sheet** — supervisor MCU lives at parent level (Teensy or ESP per `OSC-to-I2C` README) |
| C5 | I²C bus speed | 100 kHz nominal (PCA9555 standard mode ; **TBD** confirm against parent firmware) |
| C6 | SPI bus speed | up to 10 MHz per MCP42100 datasheet ; **TBD** confirm actual SCK frequency used by the show controller firmware |
| C7 | Operating temperature | 0 °C to +50 °C ambient (indoor performance / installation) |
| C8 | Output channel isolation | none assumed — common ground with parent ; per-channel TJ2242 sink only (no high-side switching, no galvanic isolation) |

## Acceptance criteria

| # | Check | Tool / threshold |
|---|-------|------------------|
| A1 | Schematic ERC clean | `kicad-cli sch erc gold.kicad_sch` → zero `severity == "error"` items (warnings expected for unresolved hierarchical ports when ERC is run on the sub-sheet in isolation — track but don't fail) |
| A2 | Output channel count | exactly 16 `VO_*` hierarchical output labels and exactly 16 × `Power_Management:TJ2242` instances ; mismatch = fail |
| A3 | Digital pot chain | exactly 8 × `Potentiometer_Digital:MCP42100` instances on the shared SPI bus (8 chips × 2 wipers = 16 setpoints, one per output channel) |
| A4 | I²C GPIO expander | exactly 1 × `Interface_Expansion:PCA9555PW` instance handling chip-select decode + INT line |
| A5 | SPI bus topology | `SCK` and `SI` (MOSI) shared bus reaching all 8 MCP42100 ; `SO` (MISO) chained or shared (SDO is open-drain on MCP42100, daisy-chain or wired-OR allowed) ; per-chip `CS` derived from PCA9555 GPIO |
| A6 | Hierarchical port set | exactly the 23 ports listed above (`SCL_2`, `SDA_2`, `SI`, `SO`, `SCK`, `CS`, `INT`, `VO_1..VO_16`) ; missing or extra ports = fail |

A1 + A2 + A3 + A4 are the **mandatory** gates for considering an
LLM-emitted candidate "spec-compliant" ; A5 and A6 are graded soft
criteria for the scoring rubric.

## Compliance

- **Grade** : prototype / pre-production hardware. No CE / FCC / UL
  certification required at this stage.
- **RoHS** : target compliance for all sourced parts ; PCA9555 /
  MCP42100 / TJ2242 are all standard RoHS catalog parts.
- **Safety classification** : SELV (5 V control side ; load side
  current per channel sub-amp). No functional safety standard claimed
  (live-arts / installation context).
- **Open-source status** : **client-confidential** project (custom
  artistic-show electronics). Internal F4L benchmarking is fine ;
  **do not** publish the schematic in external papers / public repos
  without owner approval (Guillaume Dalin / Les Amis Nos Morts).

## TBD / open questions

These fields were marked `TBD` above and need a follow-up pass before
this slot is used as a strict gold reference :

1. **C1** — exact BOM cost (read `amis nos morts PCB/jlcpcb/bom.csv`
   if present)
2. **C5** — exact I²C bus speed used by the show controller firmware
   (Teensy or ESP — see `OSC-to-I2C/` for parent firmware)
3. **C6** — exact SPI SCK frequency
4. **Inputs #1, #2** — exact logic level (3.3 V vs 5 V) used by the
   parent supervisor (Teensy ≈ 3.3 V on most variants ; ESP32 = 3.3 V ;
   MCP42100 Vdd is 2.7-5.5 V so both work)
5. **Outputs #2** — bench-measured per-channel current range (depends
   on TJ2242 external resistor + wiper voltage)
6. **Outputs #3** — confirm whether per-channel sense is wired back
   through the PCA9555 (architecturally plausible but not verified
   from the schematic alone)
7. Whether the parent `amis nos morts PCB.kicad_sch` should also be
   bundled into the corpus slot for full ERC pass (currently only
   the sub-sheet is copied — ERC will warn about unresolved
   hierarchical labels when run in isolation)
8. Hierarchical context : the sub-sheet has **no** explicit power-rail
   regulator on this sheet — assumes +5 V is provided by the parent
   board ; this is a meaningful constraint when scoring an LLM-emitted
   alternative that may try to add an on-sheet regulator

## Provenance

- Hardware design : L'Electron Rare (Clément Saillant)
- Artistic concept / client : Guillaume Dalin / Les Amis Nos Morts
  (live performance / installation art)
- Project tree : `~/Documents/Lelectron_rare/Les_Amis_Nos_Morts/`
- Parent firmware : `OSC-to-I2C/` (Teensy + ESP variants, OSC →
  I2C bridge using Adafruit PWM Servo Driver lib)
- License : project-internal — **not** open-sourced
- Open-source status : **client-confidential** (per CATALOG.md
  "Risques" section). Slot is restricted to **internal F4L
  benchmarking** only.
