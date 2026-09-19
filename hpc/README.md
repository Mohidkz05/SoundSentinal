# Training on M3 (MASSIVE)

Project `df37` — *Detecting Deepfakes Without Compromising User Privacy*.
Account granted 14 September 2026; username `mkha0155`.

## It is not a VM

This matters before anything else, because the whole workflow follows from it.
M3 is a **shared Slurm cluster**, not a machine you keep. Practically:

- **No sudo, ever.** Nothing in this directory installs a system package, and
  nothing calls `apt`. That is why the environment is built with `uv` and
  `pip` wheels rather than `module load` guesses — see `bootstrap.sh`.
- **You do not run training at a prompt.** You submit a script and it runs when
  the scheduler has a GPU free. An ssh session dying mid-run does not kill a
  submitted job; it would kill anything you started interactively.
- **Login nodes are not for compute.** The welcome email is explicit: they are
  for "light-weight account housekeeping work, including job preparation and
  submission". The one exception here is `bootstrap.sh`, which is network-bound.
- **Every job must say `--account=df37`** or the usage is not charged to the
  project and the submit is rejected. All scripts here already do.
- **Three storage areas, with different jobs.** `~/df37` is primary and backed
  up; `~/df37_scratch` is large and holds anything regenerable. `$HOME` has the
  tightest quota, so it holds the clone and nothing else. `hpc/env.sh` encodes
  this — the venv and the corpus both live outside `$HOME` deliberately.

This replaces the standalone GPU VM that was being requested from the
supervisor. M3 is strictly better for the purpose: A100-class cards rather than
a 24GB floor, and no machine to maintain.

## The five commands

```bash
# 1. From your laptop — set the cluster password first at
#    https://password-hpc.erc.monash.edu.au (independent of your Monash ID)
ssh mkha0155@m3.massive.org.au

# 2. Clone. $HOME only — see storage above. HTTPS, not SSH: M3 has no
#    deploy key for this repo, and a read-only clone needs none.
git clone https://github.com/Mohidkz05/SoundSentinal.git ~/SoundSentinal
cd ~/SoundSentinal

# 3. Build the environment (~5 min, login node is fine)
bash hpc/bootstrap.sh

# 4. Fetch the corpus (7.6GB in, ~11GB out; the slow step)
sbatch hpc/get_la.slurm

# 5. Train the non-private baseline — the first honest number
sbatch hpc/train.slurm --no-dp
```

Then `squeue -u mkha0155` to watch, and `tail -f soundsentinal-<jobid>.out`
to read. `scancel <jobid>` to stop one.

## What this account actually has

Confirmed against `mkha0155` / `df37` on 19 September 2026, so these are
measurements rather than conventions read off the docs.

**Storage** — more headroom than expected, and `$HOME` is not the squeeze it
usually is:

| Path | Quota | Used |
| --- | --- | --- |
| `/home/mkha0155` | 20 GB | 0 |
| `~/df37` (primary) | 500 GB | 0 |
| `~/df37_scratch` | 3072 GB | 190 GB |

**GPUs** — `--partition=gpu` is the one to use: `AllowAccounts=ALL`,
`AllowQos=ALL`, 40 nodes and 132 GPUs across A100-80G, L40S, A40 and T4, with a
7-day ceiling and a 1-day default.

**There is no `m3g` partition.** `m3g[100-119]` are the L40S *node names* inside
`gpu`. Worth stating because "m3g" is exactly the sort of plausible-looking
partition name that gets copied out of a half-remembered tutorial.

**H100s are reachable.** `df37` holds the `m3h` QOS, and the `m3h` partition
accepts it — four H100 nodes, 7-day limit. That is the wav2vec2 / WavLM ceiling
row in `APPROACH.md` solved, far past the 24GB VRAM floor originally requested:

```
#SBATCH --partition=m3h
#SBATCH --qos=m3h
#SBATCH --gres=gpu:H100:1
```

Don't point the 267k CNN at it — four nodes is scarce and the model cannot use
one. Plain `--gres=gpu:1` on `gpu` takes whatever frees first, which is right
for everything in the table except that last row.

**CPU jobs** default to `comp`, 7-day ceiling, so `get_la.slurm`'s 8-hour
request needs no partition named.

## How the pieces fit

| File | Does |
| --- | --- |
| `env.sh` | The paths. Sourced by everything, including your own ssh sessions. |
| `bootstrap.sh` | `uv` → venv at `~/df37/venv` → CUDA wheels. Idempotent. |
| `requirements-torch.txt` | torch + torchaudio, CUDA index only. |
| `requirements-rest.txt` | Everything else, PyPI only. Split on purpose — see its header. |
| `get_la.slurm` | ASVspoof2019 LA → `~/df37_scratch/data/LA`. Resumable. |
| `train.slurm` | `verify_setup.py`, then training. Forwards its arguments. |
| `evaluate.slurm` | Scores a checkpoint on the eval partition — the quotable number. |

Two environment variables carry the layout into the Python, so no path is
written down twice:

- **`ASVSPOOF_ROOT`** → `~/df37_scratch/data`, read by `get_corpus_paths()`.
- **`CKPT_ROOT`** → `~/df37/checkpoints`, read by `train_dp_avspoof.py`,
  `app.py` and `verify_setup.py`. Unset, all three fall back to
  `ai_model/checkpoints` and your laptop behaves exactly as before.

## Getting a trained model back

The cluster trains; it does not serve. `app.py` is a dev-server Flask app bound
to loopback and has no business on a shared node.

```bash
# from your laptop, via the data transfer node named in the welcome email
scp mkha0155@m3-dtn.massive.org.au:df37/checkpoints/nodp/best.pth \
    ai_model/checkpoints/
```

The checkpoint carries its own calibrated threshold, dev metrics and confusion
matrix, and `app.py` builds the model card from them — so the reading on
`/result` describes the run that produced it with nothing typed in by hand.

## Driving M3 from your laptop

Key-based auth, so job submission and log reading need no password. M3 offers
`publickey` with no MFA step — confirmed from its auth banner
(`publickey,gssapi-keyex,gssapi-with-mic,password`) — so this works.

`~/.ssh/m3_df37` is a key dedicated to this account, separate from the GitHub
key, so revoking either leaves the other alone. It has **no passphrase**, which
is what makes unattended use possible; the file permissions are the protection.
If you would rather have one, add it with `ssh-keygen -p -f ~/.ssh/m3_df37` and
unlock it once per session with `ssh-add`.

Install it — the one step that needs the cluster password:

```bash
ssh-copy-id -i ~/.ssh/m3_df37.pub m3
ssh m3 true && echo "key works"
```

`~/.ssh/config` defines `m3` (login) and `m3-dtn` (data transfer), both with
`IdentitiesOnly yes` so ssh does not offer the GitHub key first and exhaust the
server's retry limit before reaching the right one.

After that, everything is one-liners from the repo on your laptop:

```bash
ssh m3 'squeue -u mkha0155'                       # what is queued or running
ssh m3 'cd ~/SoundSentinal && git pull'           # ship a code change
ssh m3 'cd ~/SoundSentinal && sbatch hpc/train.slurm --no-dp'
ssh m3 'tail -40 ~/SoundSentinal/soundsentinal-*.out'
ssh m3 'scancel <jobid>'
scp m3-dtn:df37/checkpoints/nodp/best.pth ai_model/checkpoints/
```

Treat the M3 clone as **read-only**: pull, never commit. Its git identity is
unconfigured, and a commit made there is one to reconcile later for no gain.

## No internet on the compute nodes?

`get_la.slurm` checks for a route out and stops with this message if there is
none. Some clusters allow outbound HTTP only from login and data-transfer
nodes. The fallback is the DTN, inside a session that survives a dropped ssh:

```bash
ssh mkha0155@m3-dtn.massive.org.au
cd ~/SoundSentinal && source hpc/env.sh
tmux new -s la        # or: smux new-session, M3's Slurm-backed equivalent
bash hpc/get_la.slurm # the script is a plain bash script; the #SBATCH lines are comments
```

Detach with `Ctrl-b d`, come back with `tmux attach -t la`.

## State of this, as of 19 September 2026

**Verified on the cluster, not inferred:**

- Key-based ssh works; `ssh m3 true` is silent. M3 offers `publickey` with no
  MFA step.
- `bootstrap.sh` runs clean on a login node. venv at `~/df37/venv`, Python
  3.12.14, torch 2.13.0+cu130, torchaudio 2.11.0+cu130, opacus 1.6.0.
- **The GPU path works end to end.** Driver 580.126.20; a real 4096×4096 matmul
  and a `MelSpectrogram` both ran under `--partition=gpu --gres=gpu:1`, landing
  on an L40S once and an A100 80GB once. The MelSpectrogram returned
  `(1, 128, 126)` — the same shape `verify_setup.py` asserts, so the GPU path
  and the CPU path agree on the tensor the model sees.
- `get_la.slurm` submits and runs on the default `comp` partition, pulling at
  roughly 5 MB/s (~25 min for the 7.6GB).

**Two bugs this setup found in itself**, both worth knowing because the shape
recurs:

- The dependency file asked for cu129 and installed cu130 in silence. uv
  consults extra indexes before `--index-url` and stops at the first index
  holding a package *name*, never falling through for a better version. Ordering
  them the other way just moved the failure — the CUDA index also mirrors `tqdm`,
  older than the pin, and the resolution died as unsatisfiable. The fix is one
  index per file: `requirements-torch.txt` and `requirements-rest.txt`. An index
  URL is a hint, not an instruction; pin the `+cuXXX` local version and check
  with `--dry-run` **on a clean venv**, since a dry run against an
  already-correct environment passes regardless.
- `get_la.slurm` let curl write its progress meter into the job log, about two
  thousand lines of percentages burying everything else. Same defect as the
  trainer's tqdm bar, same fix.

**First trained baseline — 19 September 2026.** Job 60256177, `--no-dp`, on an
L40S: 5 epochs in **2m49s**, exit 0. Weights at
`~/df37/mkha0155/checkpoints/nodp/best.pth`, carrying a calibrated threshold of
0.5698.

Dev results by epoch, EER: 6.75% → 1.49% → 0.74% → 0.48% → **0.24%**.

**Do not quote that 0.24% anywhere.** It is *dev* EER, and the dev partition
uses attacks A01–A06 — the same six the model just trained on. Every row in
`APPROACH.md`'s comparison table is *eval* partition, A07–A19, unseen. Put
beside each other they would say this 267k-parameter 2-conv CNN beats AASIST
(0.83%), which is not a finding, it is a category error. `APPROACH.md` says this
under "Sourcing discipline" and it is the exact trap this number walks into.

One thing worth watching: the calibrated threshold moved 0.5030 → 0.6735 →
0.3342 → 0.8935 → 0.5698 across five epochs. The EER operating point is not
stable run to run, which matters more here than usual — the UI draws that
threshold on screen, so it is a number users see, and `/result` states the
relation to it in prose.

**Eval-partition result — job 60260737, 71,237 clips, under 3 minutes.**
**10.15% EER**, against 0.24% on dev: a factor of 42, and the honest version of
the dev number. It loses to both 2019 GMM baselines, which is the measured case
for AASIST. Per-attack spread runs A07 0.11% to A17 41.19%. Full analysis in
`APPROACH.md` under "What the baseline actually measured"; raw numbers in
`checkpoints/nodp/eval_eval_*.json`.

**Still not done:**

- **AASIST is not ported.** `train.slurm` trains the current 2-conv CNN, which
  the eval run below shows losing to the 2019 GMM baselines. Step 3 in "Order of
  work" in `APPROACH.md`; neither job script changes when it lands, only what
  `train_dp_avspoof.py` builds.
- **The DP arm has not been run**, so the cost-of-privacy gap — the project's
  actual research question — is still unmeasured.
- **DP arm untried on this hardware.** Opacus per-sample gradients cost 2–4× in
  memory; `--mem=32G` is a guess until a DP run is measured.
