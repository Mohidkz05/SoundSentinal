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

## Confirm these before the first train submit

Four things in `train.slurm` are written from M3's documented conventions
rather than from this account, so check them once and fix the header if they
differ. All four fail loudly at submit rather than silently, so this is
five minutes, not a risk.

```bash
user_info        # quota and current usage on the three project directories
show_cluster     # partitions, GPU types, what df37 may actually use
sinfo -s         # partition time limits — get_la.slurm asks for 8h
```

- **`--partition=m3g`** — M3's GPU partition. If `show_cluster` names something
  else for df37, change it.
- **`--gres=gpu:1`** — some GPU types need naming explicitly,
  e.g. `--gres=gpu:V100:1`.
- **`--time=08:00:00` in `get_la.slurm`** — if the default partition caps
  shorter, name a longer one.
- **The driver version**, printed by `train.slurm` at the top of every log. The
  pinned wheels are cu129, which needs a driver supporting CUDA 12 (>= 525).
  `requirements-cuda.txt` documents the cu128 fallback if it is older.

## How the pieces fit

| File | Does |
| --- | --- |
| `env.sh` | The paths. Sourced by everything, including your own ssh sessions. |
| `bootstrap.sh` | `uv` → venv at `~/df37/venv` → CUDA wheels. Idempotent. |
| `requirements-cuda.txt` | The root `requirements.txt` pins, built for CUDA. |
| `get_la.slurm` | ASVspoof2019 LA → `~/df37_scratch/data/LA`. Resumable. |
| `train.slurm` | `verify_setup.py`, then training. Forwards its arguments. |

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

## What has not been done here

Stated plainly so nothing below reads as working:

- **No job has been run on M3.** Everything in this directory is written
  against M3's documented conventions and the repo's actual entry points, and
  the syntax is checked, but the first real submit is the first test.
- **The download URL is verified, the extract path is verified.** The DataShare
  bitstream returns `application/zip`, 7,640,952,520 bytes, supports byte-range
  resume, and its first entry is `LA/ASVspoof2019_LA_asv_protocols/...` — which
  is why extracting into `$ASVSPOOF_ROOT` lands where `get_corpus_paths()`
  looks. Confirmed by range request, not assumed.
- **Still no trained weights, anywhere.** That is the point of all of this.
- **AASIST is not ported yet.** `train.slurm` trains the current 2-conv CNN.
  Step 3 in "Order of work" in `APPROACH.md`; the job script does not change
  when it lands, only what `train_dp_avspoof.py` builds.
