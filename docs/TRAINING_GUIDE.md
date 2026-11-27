# Train agricultural coverage PPO

Start with the bundled synthetic fields. After the first checkpoint works, you can continue the run on another machine or start a separate experiment with real European field boundaries.

## 1. Clone and install on Linux

Anyone can read and clone the [public repository](https://github.com/wushuaixue-boop/agricultural-coverage-ppo). GitHub's **Code → Download ZIP** also works. For the commands below, open a terminal in the extracted repository or clone it as shown.

```bash
git clone https://github.com/wushuaixue-boop/agricultural-coverage-ppo.git
cd agricultural-coverage-ppo
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.12.0+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.lock
python verify_data.py
```

Verification should end with **Verified all 12,800 fields** and report three splits. The default V1.0 backend compiles a small native library on first use and requires g++ with C++17 support. If a compiler is unavailable, append `--backend reference` to the training command. CPU is sufficient.

## 2. Train a first policy

```bash
python train.py --stage 6 --workers 2 --steps 512 --out runs/first
cat runs/first/latest.json
```

Expect a `step_512` directory with `model.zip`, `vecnormalize.pkl`, `metadata.json` and `session.json`. A smoke run checks that the installation can train, validate and save. Coverage after 512 steps is not a measure of convergence.

For a longer fresh run, use `python train.py --stage 6 --workers 4 --steps 100000 --interval 25000 --out runs/mixed`. Each run needs a new output directory. Stages stay fixed in this launcher. Stage 1 uses generated rectangles, stage 6 mixes the supplied A–D training fields with procedural fields, and stage 7 also samples robots. Choose 1, 2 or 4 workers. The rollout always contains 512 aggregate steps and requested budgets round up to a complete rollout.

### Multiple vehicles on one field

For V1.0, set `--num-vehicles` to choose how many vehicles share one field and one reward in each environment. The default is one vehicle. Set `--inter-vehicle-clearance-m` for the extra gap between vehicle bodies, which defaults to 0.5 m. Each fleet uses the same robot specification for all its vehicles, and the field must have enough room to place them safely.

```bash
python train.py --stage 1 --num-vehicles 3 --inter-vehicle-clearance-m 1.0 --steps 512 --out runs/fleet
python train.py --resume runs/fleet/step_512 --steps 512 --out runs/fleet-continued
```

One environment step advances the whole fleet. `--workers` controls the number of independent environments. The policy's action and observation shapes depend on fleet size. The checkpoint records the vehicle count and clearance, so omit both flags when resuming. The evaluator reads them from the checkpoint and reports shared raster coverage for multi-vehicle runs.

## 3. Evaluate and continue

Run these from the repository root after step 2.

```bash
checkpoint=$(python -c 'import json; print(json.load(open("runs/first/latest.json"))["checkpoint"])')
python -m ppo_coverage_planning_v1.evaluate_cli \
  --checkpoint "runs/first/$checkpoint" --dataset-manifest data/fields.json \
  --stage 6 --n-episodes 2 --output-json runs/first/evaluation.json
python train.py --resume "runs/first/$checkpoint" --steps 512 --out runs/continued
```

The evaluator writes aggregate and per-episode metrics and trajectories. Look at coverage alongside collision and clearance outcomes. This command uses validation episodes; it does not evaluate the held-out test split. The continuation should reach **1,024 cumulative steps**. Use `best.json` instead of `latest.json` to locate the checkpoint with the best validation score.

To move a run, take its entire checkpoint directory, the repository revision and the same dataset package. Supply the same `--version` and `--data` when resuming. The runner restores the optimizer, reward normalization, timestep and saved configuration. Episodes and rollout buffers restart. To view logs, run `tensorboard --logdir runs`.

## 4. Choose a dataset

| Source | Size | Use |
|---|---:|---|
| Bundled synthetic fields | 12,800 | Default for mixed-field training. Each family contributes 2,560 train, 320 validation and 320 test fields. |
| Original synthetic smoke set | 8 | Reproduce old runs with `--data data/smoke8/fields.json`. |
| Full EU source archive | 51,398,853 fields, 15.5 GB | Geographic source polygons. Download once to prepare a metric PPO dataset. |
| Registered EU subset | 94 real fields | 47 train, 17 validation, 18 test, 12 historical evaluation. Rebuild locally using step 5. |

A is convex, B is non-convex, C has a convex exterior with holes, and D has a non-convex exterior with holes. The synthetic set uses local metres. Test and historical evaluation fields are held apart from training. [Bundled dataset and generation instructions](../README.md#datasets)

## 5. Download real fields and make a training manifest

Skip this section for synthetic training. Obtain Planet and Sinergise's *EU field boundaries* from [Zenodo](https://zenodo.org/records/14229033) or [Source Cooperative](https://source.coop/planet/eu-field-boundaries/field_boundaries.parquet). Choose the original **field_boundaries.parquet**. The registered row indices refer to that file's original order. Allow at least 16 GB of free space for the download.

```bash
mkdir -p private-data
curl -fL --retry 3 -C - -o private-data/field_boundaries.parquet \
  'https://zenodo.org/records/14229033/files/field_boundaries.parquet?download=1'
echo '9275e8b3817db07bb3c96fa72f607bb7  private-data/field_boundaries.parquet' | md5sum -c -
python -m pip install pyarrow==21.0.0 pyproj==3.8.0
python tools/prepare_eu94.py private-data/field_boundaries.parquet private-data/eu94
python train.py --stage 6 --steps 512 --data private-data/eu94/real_fields_splits.json --out runs/real-smoke
```

The preparation script reads the source in batches, checks selected field IDs and original geometry hashes, projects the 94 selected polygons into their registered UTM zones, centers their metric coordinates and verifies the registered geometry hashes, areas and holes. It writes the trainer's JSON manifest and checksum file into a new directory. Keep both for continuation. This is a new serialization of the registered cohort; an older checkpoint bound to a different manifest checksum needs its original package.

Stage 6 samples eligible training records together with synthetic fields using the existing size and vehicle-feasibility rules. A short smoke run does not visit all 47 real training fields. For real-data validation, substitute the same real manifest for `data/fields.json` in step 3.

The EU dataset release on Zenodo and its publisher's README specify CC BY-SA 4.0. The archive also contains generic `planet_afb` product metadata marked proprietary. Keep the release attribution and license link with adaptations and describe the selection, classification and projection. [Release record](https://zenodo.org/records/14229033) · [Publisher’s license statement](https://data.source.coop/planet/eu-field-boundaries/README.md)

## 6. Choose the policy version

The examples above use V1.0 with two actions (target speed and curvature) and the compiled backend. To train V1.2 with curvature-only control, the untreated-area observation and anti-circling reward:

```bash
python train.py --version 1.2 --stage 3 --steps 512 --out runs/v12
(cd versions/v1_2 && python -m ppo_coverage_planning_v1.evaluate_cli \
  --checkpoint ../../runs/v12/step_512 --dataset-manifest ../../data/fields.json \
  --stage 3 --n-episodes 2 --output-json ../../runs/v12/evaluation.json)
```

Use `--version 1.2` for V1.2 continuation. The subshell selects the matching evaluator. Mixing versions changes the expected observation and action spaces.
