# GhostNet — Product Requirements Document

**Hackathon project.** WiFi-based, camera-free elderly-care + presence monitor built on two
$9 ESP32-S3 boards reading WiFi Channel State Information (CSI). Detects that a person is
present, estimates breathing/heart rate, visualizes the room, detects falls (teammate's CNN),
and places a Twilio call to a caregiver when someone falls.

Snowflake is the system of record for raw CSI and alert events (targets the "Best use of
Snowflake" track).

---

## 1. Vision / demo narrative

> Open `localhost`. You see a live view of the room sensed entirely by WiFi — a CSI heatmap
> showing motion/presence intensity, a breathing-rate trace, and an illustrative figure where a
> person is detected. No camera. When the person falls, the fall model fires, the UI raises a
> red alert, and within seconds the caregiver's phone rings (Twilio). The whole event is logged
> to Snowflake and queryable after the fact.

---

## 2. What is REAL vs ILLUSTRATIVE (read this first)

This section exists so we never make a claim on stage we can't defend. WiFi CSI at 2.4 GHz from
two boards is genuinely good at *motion/presence* and *coarse vitals*; it is **not** a camera and
cannot localize a body in 2D or produce a real skeleton.

| Feature | Status | Honest framing for the pitch |
|---|---|---|
| **Presence / motion** | ✅ Real | "Detects whether the room is occupied and whether the person is moving." This is CSI's strongest signal. |
| **Breathing rate** | ⚠️ Real pipeline, unvalidated | "Estimates breathing rate from chest-motion modulation." Say *estimate*, not *medical*. |
| **Heart rate** | ⚠️ Weak | Only credible when the person is still. Treat as best-effort; consider hiding if noisy. |
| **CSI heatmap** (subcarrier × time, per board) | ✅ Real | This is actual radio data off the boards — the most honest "WiFi sensing" visual. |
| **Room position heatmap** (where in the room) | ❌ Not real with 2 boards | True 2D localization needs more receivers + tomography. Do NOT claim it. We can show per-board *motion intensity*, not a position. |
| **Skeleton / pose** | ❌ Illustrative only | CSI→pose at 2.4 GHz is ~2.5% PCK (effectively noise — the model authors say so). Render a **placeholder figure keyed to presence/motion**, clearly labeled "illustrative." |
| **Fall detection** | 🔨 Teammate's CNN | The only viable fall path. The old bbox heuristic in `backend.py` is dead (see §6). |
| **Caregiver call** | ✅ Real (Twilio) | Real outbound voice/SMS on fall. |

**Banned claims:** "100% presence accuracy" (the model authors publicly retracted this — it was
measured on a single-class recording), any fall-detection accuracy %, "tracks where people are,"
"medical-grade vitals."

---

## 3. Architecture (current, post-pivot)

We are **dropping Docker and the RuView Rust sensing server.** It synthesized a fake skeleton and
fabricated bounding boxes; `fake_server.py` only mocked events. New pipeline is pure Python +
the pretrained HF model + the teammate's fall model.

```
  ESP32-S3 #1 ─┐
               ├── UDP :5005 (magic 0xC5110001, 64 subcarriers, I/Q int8)
  ESP32-S3 #2 ─┘            │
                            ▼
                 csi_receiver.py  ──►  Snowflake  csi_raw   (archival, every frame)
                            │
                            ├──► feature_extractor   (CSI window → 8 features)   [NEW]
                            │            │
                            │            ├──► HF encoder + presence head  → presence, embedding
                            │            └──► vitals (FFT)                → breathing/heart rate
                            │
                            └──► fall_model (teammate's CNN)  → {fall, confidence}   [TEAMMATE]
                            │
                            ▼
                        backend.py  (FastAPI)
                            │  - merges presence / vitals / fall into events
                            │  - debounce + fall-confirm window
                            │  - writes alert events → Snowflake  events
                            │  - on confirmed fall → Twilio call   [NEW]
                            │
                ws://localhost:8000/ws  │  GET /events  │  GET /csi (heatmap feed)  [NEW]
                            ▼
                  React + Vite frontend (existing)
       CSI heatmap │ presence badge │ vitals trace │ illustrative figure │ fall alert │ event log
```

### The model integration reality (important)
The HF model `ruvnet/wifi-densepose-pretrained` (a.k.a. `ruv/ruview`) is **not** a raw-CSI →
position model. It is an **8-dim feature → 128-dim embedding encoder + a presence head**. To use it:

1. `huggingface-cli download ruvnet/wifi-densepose-pretrained --local-dir models/`
   (Note: the model card's quick-start says `ruv/ruview`; the canonical repo id is
   `ruvnet/wifi-densepose-pretrained`. Verify which resolves before relying on it.)
2. Build a **feature extractor** (does not exist in either repo — this is new work) that turns a
   CSI window into the 8 features the encoder expects: signal-disturbance, motion rate, breathing
   band energy, heart band energy, phase variance, person-count proxy, fall proxy, RSSI.
   - Several of those 8 (presence, fall, person_count) are *semantic outputs*, so feeding them as
     inputs is partly circular. **Realistic plan:** compute the directly-measurable ones
     (amplitude/phase variance, motion rate, breathing/heart band energy via FFT, RSSI) and use the
     model's **presence head** as the headline "model-backed" result; treat the embedding as a
     nice-to-have. Do not over-promise the model doing localization or pose.
3. The repro/training script referenced by the model card (`aether-arena/staging/train_csi_embed.py`)
   is **absent** from this clone — we cannot retrain or reproduce the 82.3% number locally. Use the
   published weights as-is.

If wiring the model proves fiddly under time pressure, presence is fully achievable with plain CSI
amplitude-variance thresholding (no model) — keep that as the fallback.

---

## 4. Components & ownership

| Component | Owner | Status | File(s) |
|---|---|---|---|
| ESP32-S3 firmware / CSI UDP emit | (shared) | ✅ Reads CSI; accuracy unverified | (firmware) |
| `csi_receiver.py` — UDP→Snowflake | you/team | ✅ Working | `csi_receiver.py` |
| Feature extractor (CSI→8 feat) | **you** | 🔨 New | `features.py` (new) |
| HF model load + presence/embedding | **you** | 🔨 New | `model.py` (new) |
| Vitals (FFT breathing/HR) | **you** | 🔨 New (port from RuView `vital_signs.rs`) | `vitals.py` (new) |
| `backend.py` — merge/debounce/Twilio | **you** | ⚠️ Exists; rewire off Docker; add Twilio | `backend.py` |
| CSI heatmap feed + endpoint | **you** | 🔨 New | `backend.py` + frontend |
| Frontend visualizer | **you** | ⚠️ Exists; add heatmap + figure | `frontend/src/` |
| Fall model (CNN on CSI) | **teammate A** | 🔨 In progress | `fall-detect/` (currently empty) |
| Snowflake schema + analytics | **teammate B** | ⚠️ `csi_raw`/`events` in use | (Snowflake) |
| Twilio caregiver call | **you** | 🔨 New | `backend.py` + `.env` |

---

## 5. Visualizer spec (your piece)

- **CSI heatmap** — per board, subcarrier index (y) × time (x), color = amplitude. Real data,
  streamed from `backend.py`. This is the centerpiece "look, it's WiFi" visual.
- **Presence badge** — occupied / empty, from the model's presence head (fallback: amplitude
  variance threshold). Show the ~30 s calibration state on startup.
- **Vitals trace** — breathing rate (and heart rate if stable). Already exists (`VitalsGraph.jsx`).
- **Illustrative figure** — a simple silhouette/skeleton that appears when present and animates with
  motion level. **Must be labeled "illustrative — not a measured position."** Driven by presence +
  motion scalar, not by real localization.
- **Fall alert** — full-screen red banner + the caregiver-call status. Already exists
  (`FallAlert.jsx`); wire it to the real fall event and the Twilio call result.
- **Event log** — already exists (`EventLog.jsx`).

---

## 6. Decisions locked

1. **No Docker, no RuView Rust server, no `fake_server.py`** in the demo path (fake_server stays
   only as an offline UI test harness).
2. **Fall detection = teammate's CNN only.** Delete/disable `is_fallen()` and the `persons[].bbox`
   path in `backend.py` — that bbox was synthetic and the heuristic (`width/height >= 1.0`) can
   never fire on real data.
3. **Skeleton is illustrative**, presence/vitals/heatmap are the real outputs.
4. **Snowflake stays** as system of record (`csi_raw` + `events`).

---

## 7. What you're missing / risks to plan for

- **Feature extractor is net-new work** — the model can't eat raw CSI; budget time for `features.py`.
  Have the no-model amplitude-variance presence fallback ready.
- **Two boards can't localize.** Set the heatmap expectation to "motion intensity," not "position."
- **UDP :5005 contention** — `csi_receiver.py` and any other consumer both want port 5005. Run a
  single receiver that fans out (archive to Snowflake *and* feed the pipeline), or use two ports.
- **Calibration / drift** — presence needs ~30 s empty-room baseline; a noisy hackathon venue
  (hundreds of APs, people walking past) will drift it. Plan a re-baseline button.
- **Boards must sniff the right channel** — both ESP32s and the monitored traffic must be on the
  same WiFi channel, or CSI is sparse. Verify before the demo.
- **Twilio logistics** — trial accounts can only call *verified* numbers; needs Account SID, Auth
  Token, a Twilio number, and the caregiver number. Store in `.env` (never commit). Pre-verify the
  demo phone.
- **False-alarm guard on the call** — require N consecutive fall frames + a short "are you ok?"
  cancel window before dialing, so a flicker doesn't spam the caregiver mid-demo.
- **Fall model integration contract** is undefined — agree the interface with teammate A now
  (see §8).
- **Hardware accuracy unknown** — "CSI reads successfully but unsure how accurate the boards are."
  Do a sanity capture: empty room vs. someone waving should show an obvious amplitude-variance
  difference. If it doesn't, fix the boards before building on top.

---

## 8. Fall-model integration contract (for teammate A)

Define so the CNN drops in cleanly:

- **Input:** a rolling CSI window — `N` frames × `64` subcarriers of amplitude (and/or phase) per
  board. Agree `N` (e.g., 2–3 s ≈ 50–70 frames at ~24 fps) and whether one or both boards.
- **Output (per inference):** `{ "fall": bool, "confidence": float }` at a fixed cadence.
- **Delivery:** either (a) a Python function `predict(window) -> dict` that `backend.py` calls in
  the pipeline, or (b) the model process posts to an internal endpoint / queue that `backend.py`
  consumes. Prefer (a) for the hackathon.
- `backend.py` applies the debounce + confirm window, emits `fall_detected`, logs to Snowflake, and
  triggers Twilio.

---

## 9. Milestones (suggested order)

1. **Sanity-check the boards** — empty vs. occupied amplitude variance is clearly separable.
2. **Single fan-out receiver** — one process: UDP :5005 → Snowflake `csi_raw` + in-memory window.
3. **Presence (no model)** — amplitude-variance threshold → `presence_update`. Get the end-to-end
   UI lighting up on real data.
4. **CSI heatmap** in the UI from real frames.
5. **Vitals** — FFT breathing rate trace.
6. **HF model** — load weights, feature extractor, swap presence to model-backed (keep fallback).
7. **Fall model** — integrate teammate's CNN per §8 contract.
8. **Twilio** — confirmed-fall → caregiver call, with false-alarm guard.
9. **Snowflake analytics** — a query/dashboard over `csi_raw` around `events` for the track.
10. **Demo rehearsal** — on the venue WiFi, with the re-baseline button.

---

## 10. Snowflake (track: Best use of Snowflake)

- `csi_raw(timestamp, board_id, rssi, subcarriers, amplitudes VARIANT)` — every frame (already wired).
- `events(event, timestamp, confidence, heart_rate_bpm, breathing_rate_bpm)` — alerts (already wired).
- **Pitch angle:** historical/forensic analytics — query CSI patterns in the window around a
  confirmed fall, trend breathing rate over a night, occupancy heatmap over time. The value is "raw
  sensor truth, queryable after the fact," not just storage.

---

## 11. Demo plan — record & replay (the actual demo we're shipping)

Because 2 single-antenna boards cannot compute a real 2D position, the demo uses **pre-recorded
real CSI replayed live**, with an **authored trajectory** driving a blob in a 3D room view. This is a
legitimate demo technique: the radio data is real, the path is illustrative.

### Flow
1. **3D space visualizer** — a room rendered in 3D (Three.js / react-three-fiber): floor, walls
   (semi-transparent), the two board positions, and a moving **blob** (person).
2. **Pre-record (before demo):**
   - **"Walk-around" capture** — film/record CSI while you run around the room. Save the
     `*.csi.jsonl` (same format already used: magic `0xC5110001`, 64 subcarriers).
   - **"Fall" capture** — a separate recording where you fall.
   - Optionally also film yourself on camera for the slide that proves "this CSI = me moving."
3. **During demo:** a **replay server** streams the recorded CSI frames at their original timing
   into the live pipeline → the CSI amplitude heatmap animates from **real data**, and the blob
   walks the authored path in sync.
4. **Through-walls moment:** in the 3D scene, place a wall between a board and the blob's path; the
   blob stays visible/tracked while "behind" the wall. Honest basis: WiFi genuinely penetrates
   drywall, so motion is still detected when occluded — demonstrate that the **CSI variance stays
   elevated while you're behind the wall** (real), even though the exact position is scripted.
5. **Fall:** replay the fall recording → fall fires (teammate's CNN if ready, else a scripted fall
   marker at the known timestamp) → red alert → **Twilio call to your phone**.

### How to "mock" it honestly (the technique)
- **Replay is real CSI**, not fabricated — distinct from the retired `fake_server.py`.
- **Authored trajectory file** drives the blob: a JSON time-series synced to the recording, e.g.
  ```json
  [
    { "t": 0.0,  "x": 1.0, "y": 0.0, "z": 1.0, "behind_wall": false },
    { "t": 2.5,  "x": 3.2, "y": 0.0, "z": 1.4, "behind_wall": true  },
    { "t": 8.0,  "x": 2.0, "y": 0.0, "z": 0.5, "fall": true }
  ]
  ```
- **Make it more defensible (recommended):** couple the blob's **motion magnitude/speed to the real
  CSI variance** of the replayed frames, while the **path is authored**. Then "the person is moving
  *now*" is driven by real radio; only "*where*" is illustrative. This is a much stronger answer to a
  judge than a fully scripted animation.

### Honesty / judge Q&A prep
- "Is the position computed from WiFi?" → *"The CSI is real and streamed live; the **motion** is
  derived from the real signal. The **path** is authored — true 2D localization needs a multi-antenna
  array or a mesh of nodes, which is the next hardware step. Two $9 boards reliably give presence,
  motion, vitals, and falls."*
- Don't claim computed coordinates, pose accuracy, or "100% accuracy."

### Demo scenes (scripted)
- **Scene 1 — tracking + through-wall:** blob runs in a **circle** around the room, then **walks
  out the door behind the wall** and is **still sensed** (motion variance stays elevated but
  attenuated while `behind_wall`). Loops.
- **Scene 2 — fall vs. lie-down:** person **lies down slowly → NO alert** (gradual descent, low
  impact). Then later **falls → ALERT** (sudden descent + stillness, high impact) → Twilio call to
  your phone. The lie-down-first beat is the point: the system doesn't false-alarm on normal motion.
- Then the team presents the **Snowflake** story (the judged track).

Implemented by `mock_csi.py`: a synthetic generator that streams real-format frames (synthetic CSI
whose variance tracks motion) + authored position + a believable impact/acceleration metric that
separates lie-down from fall. Scene-switchable for stage control.

### Components for the demo
| Piece | Status | File |
|---|---|---|
| `replay.py` — stream a `*.csi.jsonl` at real cadence to the pipeline (UDP or directly to backend) | 🔨 New | `replay.py` |
| Authored trajectory JSON (walk + fall) | 🔨 New (author by hand to match recording) | `demo/track-walk.json`, `demo/track-fall.json` |
| 3D room visualizer + blob + walls | 🔨 New | `frontend/src/` (Three.js / react-three-fiber) |
| CSI variance → blob speed coupling | 🔨 New | `backend.py` + frontend |
| Fall trigger (CNN or scripted marker) | 🔨 New / teammate | `backend.py` |
| Twilio call to your phone | 🔨 New | `backend.py` + `.env` |

### Demo risks
- **Replay/track drift** — author the trajectory against the recording's timestamps and lock the
  replay to original timing so the blob and CSI stay in sync; rehearse the exact run.
- **Twilio** — pre-verify your phone number (trial accounts only call verified numbers); test the
  call before going on stage; add a manual "trigger fall" key as a backup if the model/marker misses.
- **Loop cleanly** — let the replay loop so a delayed demo slot still shows motion.
- Keep one **fully-canned fallback video** of the working demo in case live replay fails.

---

## 12. Snowflake (track: Best use of Snowflake)

(Renumbered from §10.) `csi_raw` + `events` as system of record; pitch = forensic/historical
analytics over real CSI around fall events, breathing trends overnight, occupancy over time.

---

## 13. Explaining the math to judges

- **CSI** = a complex number per subcarrier: 64 frequencies, each with amplitude + phase. From the
  ESP32's raw I/Q, amplitude = √(I² + Q²).
- **Presence / motion** = **variance** of amplitudes over a short window. Empty room = low variance;
  a moving body scatters the radio paths = high variance. (This is the blob's `motion` value.)
- **Breathing / heart rate** = a chest rising/falling modulates the signal *periodically*. Run an
  **FFT** on the amplitude time-series; peak in 0.1–0.5 Hz = breathing (6–30/min), ~0.8–2 Hz =
  heartbeat. Peak frequency × 60 = the rate.
- **Fall** = a **sudden spike** in variance (whole body moving fast) **followed by stillness**.
  Lying down is gradual → no spike → no alert. Detect impact, then require ~1 s of stillness before
  alerting. This is exactly the lie-down-vs-fall distinction in the demo.
- **Why no position/skeleton** = locating (x,y) needs angle-of-arrival from an **antenna array** or
  many crossing links (tomography). Two single-antenna $9 boards give *how much* motion, not *which
  direction* — so the figure's path is the captured walk replayed, an honest visualization.

**Model vs math, say it plainly:** presence/motion/breathing = signal math (no model); fall = a small
CNN; position = neither (replayed capture). Pitch line: *"We recreated DensePose-From-WiFi (antenna-
array routers) on two $9 chips — couldn't recover limbs/pose, but we get presence, motion, breathing,
and falls, archived in Snowflake."*

## 14. Demo file structure & run commands

CSI data files live in `data/` (baked, committed); streamers play them into the visualizer.

| File | What it is |
|---|---|
| `demo/walk-session.json` | Authored walk path (waypoints) — edit to match your walk video |
| `tools/gen_csi.py` | Bakes the scenes → `data/*.csi.jsonl` (run once; re-run after editing the path) |
| `data/walk-csi.jsonl` | Captured walk CSI session (480 frames / 24 s, loops) |
| `data/fall-csi.jsonl` | Captured fall CSI session (440 frames / 22 s): stand → lie-down → up → fall |
| `demo_stream.py` | Shared engine: replays a session file over `ws://localhost:8000/ws`, runs fall detection + Twilio |
| `csi_position.py` | **Walking demo** — streams the walk session (no alerts) |
| `csi_fall.py` | **Fall demo** — streams the fall session, fires the alert + caregiver call |
| `mock_csi.py` | All-in-one live generator with scene-switch (`/scene/walk`, `/scene/fall`) — alternative to the two scripts |

**Run the demo:**
```bash
# regenerate data files only if you edited the walk path
python tools/gen_csi.py

# Scene 1 — walking / tracking / through-wall
python csi_position.py          # ws://localhost:8000/ws
cd frontend && npm run dev      # http://localhost:5173

# Scene 2 — fall (lie-down is silent, the fall alerts + calls)
python csi_fall.py
curl localhost:8000/trigger/fall   # stage backup: force a confirmed fall
```

### When does it actually call? (two-stage, like real systems)
You never call on impact — they might just get up. The flow:
1. **Impact + ~0.8 s stillness** → confirmed *fall* → on-screen alert + **countdown starts** (not a call yet).
   A gradual lie-down never spikes the impact, so it never reaches this stage.
2. **Grace countdown** (`FALL_GRACE_S`, default **5 s** for the demo; ~30 s in production) → if they
   **move / get up**, it **cancels** (`fall_cancelled`, no call). If they **stay down** the whole
   countdown → **place the caregiver call** (`call_placed`).
The frontend shows "Calling caregiver in Ns — I'm Okay / Cancel", then "Caregiver Called". Verified:
lie-down silent, fall → detected, call fires exactly one grace-period later; get-up during grace cancels.
Set the window with `FALL_GRACE_S=3 python csi_fall.py`.

**Demo-video beat (fall scene):** one person lies down → **no alert** (gradual, no impact spike) →
gets up → falls → after ~1 s of stillness the alert fires, a countdown runs, and if they stay down
the **Twilio call** goes out. Real
Twilio needs `pip install twilio` + `TWILIO_ACCOUNT_SID/AUTH_TOKEN/FROM` + `CAREGIVER_TO` in `.env`
(pre-verify your number on a trial account); otherwise it logs `[MOCK CALL]`.

---

*Honesty contract: replayed CSI + heatmap + motion magnitude are real; the blob's path and the
skeleton are illustrative; fall accuracy is whatever the teammate's CNN achieves. The demo is a
record-and-replay — that's a normal demo technique, just never claim the position is computed.*
