# Transformer - v7 (AlphaQubit)

This subdirectory is our seventh iteration of recreating AlphaQubit — Google DeepMind's
transformer-based quantum error correction decoder described in their 2024 Nature paper.
It is not fully tested or finalized, but it is the most complete and faithful version we
have built so far.

## What's new in v7

**Architecture:** The convolutional layers now use the per-distance dilation schedules
from Table S4 of the AlphaQubit paper (e.g., `[1,1,1]` for d=3, `[1,1,2]` for d=5),
and the learning rates are set per-distance from Table S3. Previous versions used
hand-tuned values that didn't match the paper.

**Training infrastructure:** The model can now train across multiple GPUs using
PyTorch DistributedDataParallel (DDP) via `torchrun`. This was necessary to get
training times into a reasonable range on the ARC HPC cluster. Single-GPU training
on a desktop 3080Ti was estimated at 200–300 hours for a full run.

**Data:** We now train on the fixed dataset created by Tzu-Chen, which
gives us a trusted, reproducible baseline rather than freshly generated simulation data.
The dataset covers surface codes at multiple distances (d=3, d=5) and round counts,
in both X and Z basis.

**Storage:** All training data is packed into a single `.h5` file (`pretrain.h5`) using
h5py. An HDF5 file acts like a compressed filesystem-in-a-file — all 130 surface code
subdirectories and their shots are embedded as datasets inside one 157MB file. This
makes copying data to cluster nodes fast and keeps I/O manageable during training. Currently, `pretrain.h5` is the only `.h5` file to be made, but once we move into the finetuning and testing stage, more `.h5`'s will be created and used.

---

## How to run locally

> **Heads up:** Local training is not really practical for a full run. Even on a desktop
> with a 3080Ti and NVMe storage it takes 200–300 hours. Running locally is fine for
> testing that the pipeline works end-to-end, but for actual results you'll want the ARC.

### 1. Get the data into place

Make sure the Tzu-Chen dataset is somewhere under the `data/` directory. It doesn't
matter what subdirectory it's in — the setup script will find it automatically. Your
`data/` folder might look something like:

```
data
├───data 0301   ← could be here
├───fun data
│   └───data 0301   ← could be here
├───other things
│   └───fun things
│       └───data
│           └───data 0301  ← could be here
    ...
```

### 2. Move the surface code directories

```bash
python src/trans7_alphaqubit/move_surface_code_dirs.py
```

This scans `data/` for all `surface_code_b*` directories from the Tzu-Chen dataset and
moves them into `data/trans7_data/`. This step is also done automatically if you skip
straight to step 4.

### 3. Build `pretrain.h5`

```bash
python data/data_random_sample.py
```

This reads all the surface code directories in `data/trans7_data/`, randomly samples
shots from each, and writes everything into a single `data/trans7_data/pretrain.h5` file
with a pre-shuffled index. Again, for right now we are only creating a pretrain file, but once we step into the finetuning and testing stages, more `.h5`'s will be created.

The model can't train on arbitrary mixed batches (a batch of d=3 r=1 and d=3 r=25 shots
can't be stacked into a single tensor since they have different sizes), so the HDF5 file
pre-groups samples by `(distance, rounds)` so the dataloader can always form
shape-consistent batches. This step is also done automatically by step 4 if the file
doesn't exist yet.

### 4. Run pretraining

```bash
python src/trans7_alphaqubit/run_pretrain.py
```

Single-GPU pretraining. Checkpoints are saved to `checkpoints/pretrain/` whenever
validation LER improves. For multi-GPU local training, use `run_pretrain_ddp.py` with
`torchrun` instead (same as the ARC workflow below).

---

## How to run on the ARC

> This assumes you already have ARC access and can log into a login node (e.g.,
> `tinkercliffs2.arc.vt.edu`). If you don't have that yet, you'll need to sort that out
> first — this section won't be much help otherwise.

Also make sure you've completed at least through step 3 of the local setup so that
`pretrain.h5` exists before continuing here.

### 1. Set up the directory structure

The SLURM scripts expect a specific layout under your home directory on ARC. Log in
and create the following:

```bash
mkdir -p ~/src/trans7_alphaqubit
mkdir -p ~/src/data/trans7_data
```

Your home directory on ARC should end up looking like this:

```
~/ (home)
├── src/
│   ├── trans7_alphaqubit/    ← model code and SLURM scripts go here
│   └── data/
│       └── trans7_data/
│           └── pretrain.h5   ← the only data file you need to transfer
```

The conda environment and its packages get installed automatically by the SLURM scripts
into `/scratch/$USER/` (your scratch space), not your home directory. Scratch has much
more storage and faster I/O, which is why the environment lives there rather than in `~`.

### 2. Transfer files to ARC

You'll need to move two things from your local machine to ARC: the `pretrain.h5` data
file and the `trans7_alphaqubit` code directory. You can't "right click + paste" to a
remote server — you use `scp` (secure copy) from a terminal on your local machine.

**Transfer the data** (157MB, takes a minute or two):

```bash
scp data/trans7_data/pretrain.h5 {USER}@tinkercliffs1.arc.vt.edu:~/src/data/trans7_data/
```

**Transfer the code** (the `-r` flag means recursive — copies the whole directory):

```bash
scp -r src/trans7_alphaqubit {USER}@tinkercliffs1.arc.vt.edu:~/src/
```

Alternatively, if the repo is on GitHub you can just `git clone` it directly on ARC
and then only transfer `pretrain.h5`, which is cleaner if you're making frequent code
changes.

> **Note:** Replace `{USER}` with your own ARC username.

### 3. Run the smoke test first

Before committing to a 168-hour training job, always run the smoke test. It runs 100
steps on 2 GPUs, takes under 30 minutes, and will catch most problems before they waste
a week of queue time.

Navigate to the code directory on ARC:

```bash
cd ~/src/trans7_alphaqubit
```

Then submit the smoke test:

```bash
sbatch run_trans7_pretrain_test.slurm
```

`sbatch` submits the job to the SLURM queue and returns a job ID immediately. The job
will start whenever the requested resources (2 GPUs, 16GB RAM) become available. To
check if your job is queued or running:

```bash
squeue -u $USER
```

Once it starts, you can watch the output live:

```bash
tail -f logs/trans7_test_<JOB_ID>.out   # SLURM script output (setup steps, env install)
tail -f logs/trans7_test_<JOB_ID>.err   # Python/model output (training progress, errors)
```

The `.out` file is everything the bash script prints (conda setup, data copy, etc.).
The `.err` file is everything Python prints — the training progress bar, loss values,
and any tracebacks if something crashes. You'll spend most of your time watching `.err`.

Both files appear in `logs/` inside the `trans7_alphaqubit` directory on ARC.

### 4. Run the full training

If the smoke test passed cleanly, you're ready for the real thing:

```bash
sbatch run_trans7_pretrain_ddp.slurm
```

This requests 2 H200 GPUs for up to 168 hours (the ARC's maximum job time). The full
training is 1,000,000 steps and takes roughly 135–160 hours on 2 H200s, so it should
finish within a single job. Monitor it the same way as the smoke test:

```bash
squeue -u $USER                                   # check if running
tail -f logs/trans7_ddp_<JOB_ID>.err             # watch training progress
```

A rough progress guide if the job takes the full 168 hours:

| Hours elapsed | Steps completed | % done |
|---------------|-----------------|--------|
| 20 h | ~119k | 12% |
| 50 h | ~298k | 30% |
| 84 h | ~500k | 50% |
| 120 h | ~714k | 71% |
| 168 h | 1,000k | 100% |

Checkpoints are saved to `~/src/trans7_alphaqubit/checkpoints/pretrain/` whenever
validation LER improves. There are two checkpoint files per run:

- `pretrain_x+z_d3.pth` — best model so far (weights only, used for evaluation)
- `pretrain_x+z_d3_resume.pth` — full training state including optimizer momentum,
  current step, and history (used to resume if the job is interrupted)

### 5. Auto-requeue (if training doesn't finish in one job)

The SLURM script handles this automatically. 120 seconds before the time limit, ARC
sends the job a warning signal. The script catches it and immediately submits the next
job with the resume checkpoint path already filled in. The new job starts as soon as
resources are available after the current one ends.

You don't need to do anything — just check `squeue -u $USER` after the first job ends
to confirm the next one is queued. If for some reason the auto-requeue didn't fire (e.g.,
no resume checkpoint existed yet because training crashed before the first eval at step
15k), you can requeue manually:

```bash
sbatch --export=ALL,RESUME_CKPT=~/src/trans7_alphaqubit/checkpoints/pretrain/pretrain_x+z_d3_resume.pth \
    run_trans7_pretrain_ddp.slurm
```

### 6. Potential problems

Here's what the scripts do on startup and where things tend to go wrong:

1. **Conda environment check** — the script checks for Miniforge3 in
   `/scratch/$USER/miniforge3`. If it's not there, it downloads and installs it. If it's
   there but broken (this has happened — a package called `boltons` sometimes goes
   missing from the base env), it attempts a minimal repair before continuing.

2. **Dependency install** — if `$ENV_PREFIX/.deps_installed` doesn't exist, it installs
   PyTorch and everything in `requirements_transformer.txt`. This only runs once; after
   that the flag file skips it.

3. **Data copy** — `pretrain.h5` is copied from your home directory to `/tmp/` on the
   compute node before training starts. `/tmp/` is local NVMe on the node and much
   faster than reading from the network filesystem during training. If the copy fails
   (e.g., the source path is wrong), the job exits here.

4. **Training** — torchrun launches the model on all requested GPUs and training begins.

The two most common failure points are **step 1** (broken conda, usually fixable by
waiting for other jobs to finish and then resubmitting) and **step 3** (wrong path to
`pretrain.h5`, check that the file is at `~/src/data/trans7_data/pretrain.h5`).

If something goes wrong, the full error will be in the `.err` log. Most errors are
fixable — read the message, don't panic.
