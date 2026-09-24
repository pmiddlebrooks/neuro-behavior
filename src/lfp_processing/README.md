# Neuropixels 1 LFP processing

Convert Open Ephys `np1-lfp_*.raw` files into one averaged LFP trace per brain area, saved next to the session’s spike-sorting files as `lfp.mat`.

Raw files live in `E:/lfp_data` (`paths.lfpDataPath`). Output goes in the same folder as `spike_times.npy` / `cluster_info.tsv`.

## Layout

```
lfp_processing/
├── run_process_np1_lfp.m            # Set sessionType / subject / session, then run
├── process_np1_lfp.m                # Main pipeline
├── find_np1_lfp_files.m             # Match and order raw files by session date
├── parse_session_recording_date.m   # ey9166_2026_04_09 -> 2026-04-09
├── resolve_lfp_session_folder.m     # Spike session folder for this task
├── load_np1_lfp_channel_depths.m    # Channel IDs and surface-referenced depths
├── select_area_lfp_channels.m       # Two channels per area (1/3 and 2/3 depth)
├── read_np1_lfp_samples.m           # Chunked read across stitched files
└── np1_lfp_constants.m             # NP1 LFP sample rate, gain, output name
```

## Usage

Set `sessionType`, `subjectName`, `sessionName`, and `brainAreas` in `run_process_np1_lfp.m`, then run that script.

```matlab
sessionType = 'interval';           % 'spontaneous' or 'interval'
subjectName = 'ey9166';
sessionName = 'ey9166_2026_04_09';
brainAreas = {'M23', 'M56', 'DS', 'VS'};  % also available: 'CC'

process_np1_lfp(sessionType, subjectName, sessionName, 'brainAreas', brainAreas);
```

Or from the command window after those variables are in the workspace:

```matlab
process_np1_lfp
```

Existing `lfp.mat` is left in place unless you pass `'overwrite', true`.

### Optional arguments

| Name | Default | Meaning |
|---|---|---|
| `brainAreas` | `{'M23','M56','DS','VS'}` | Areas to store |
| `overwrite` | `false` | Rebuild `lfp.mat` |
| `lfpDataPath` | `paths.lfpDataPath` | Folder of `np1-lfp_*.raw` |
| `outputFolder` | spike session folder | Where to write `lfp.mat` |
| `maxDurationSec` | `[]` (full recording) | Process only the first N seconds |

## What it does

1. **Match files by date.** Session `ey9166_2026_04_09` selects `np1-lfp_2026-04-09T*.raw`, including segmented names such as `np1-lfp_2026-03-24T18_56_33_0_to_2_hrs.raw`.
2. **Stitch splits.** Same-day files are concatenated in timestamp order, then by segment start hour (`_0_to_2_hrs` before `_2_to_4_hrs`). If `params.py` names an `np1-spike_` file with that timestamp, earlier aborted takes from the same day are skipped so LFP lines up with spikes. A truncated last frame (file size not a multiple of 384 channels) is dropped with a warning.
3. **Pick two channels per area.** Depth uses the same convention as `load_session_cluster_info.m`: `depth = 3840 - phy_y`, so **0 is the surface (M23)** and **3840 is deepest (VS)**. Area bounds come from `brain_area_depths.mat` when present, otherwise the defaults in `get_brain_area_depth_ranges.m`. For each area, the two raw channels nearest **1/3** and **2/3** of that depth span are averaged. The NP1 reference site (channel 191) is not used.
4. **Filter and downsample.** Convert to µV, low-pass below 300 Hz, resample 2500 Hz → 1000 Hz.
5. **Save** `lfp.mat` in the spike session folder.

Raw NP1 LFP is 384 channels, uint16, 2500 Hz (Open Ephys Neuropixels plugin). Processing is chunked so the 10–20 GB binaries are never fully loaded.

## Output (`lfp.mat`)

| Variable | Contents |
|---|---|
| `lfpPerArea` | `nSamples × nAreas`, `single`, µV, 1000 Hz |
| `lfpMeta` | Sampling rates, source files, area names, channel IDs, channel depths, depth ranges |

Load with:

```matlab
load(fullfile(sessionFolder, 'lfp.mat'), 'lfpPerArea', 'lfpMeta');
```

A 2-hour session with 4 areas is about 120 MB (`single`). That is why this pipeline writes `.mat` rather than CSV.

### Default depth ranges (µm from surface)

Used when the session has no `brain_area_depths.mat`:

| Area | Range |
|---|---|
| M23 | 0–500 |
| M56 | 501–1240 |
| CC | 1241–1540 |
| DS | 1541–2700 |
| VS | 2701–3840 |
