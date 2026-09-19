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
