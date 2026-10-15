# dlwpt-code-2e

Code for the book Deep Learning with PyTorch, Second Edition (Huang, Stevens, Antiga, Viehmann, Manning), one folder per chapter, mostly Jupyter notebooks. Leo cloned it on 2026-10-12 to learn PyTorch and works through the chapters in his own `ch*/` notebooks.

## Repo
- `origin`: https://github.com/tyrpakleo-extropic/dlwpt-code-2e (Leo's fork), branch `main`.
- `upstream`: https://github.com/deep-learning-with-pytorch/dlwpt-code-2e (the book authors' repo).
- Sync (local, git-excluded `SYNC.md`): `git pull --rebase origin main`, `git push origin main`. Book updates: `git fetch upstream && git merge upstream/main`.

## Start here
- `README.md`: book description and the upstream environment setup (Poetry or pip, torch installed separately).
- `p2_run_everything.ipynb`: runs the part 2 training apps (`p2ch13`, `p2ch14`, ...) in sequence through a `run(app, *argv)` helper.
- `/mnt/data0/shared/leot/AGENTS.md` (entry for `dlwpt-code-2e/`): env and kernel rules for this repo.

## Layout
| Path | Purpose |
|---|---|
| `p1ch2/` | Pretrained networks, CycleGAN, inpainting, model zoos, MNIST |
| `p1ch3/` | Tensors, named tensors |
| `p1ch4/` | Real data as tensors: image, volumetric CT, tabular wine, time series bikes, text, video, audio |
| `p1ch5/` | Parameter estimation, autograd, optimizers |
| `p1ch6/` | Neural networks, activation functions, `nn.Module` subclassing |
| `p1ch7/` | Datasets, birds vs airplanes (CIFAR-10) |
| `p1ch8/` | Convolutions |
| `p2ch9/` | Tokenization, attention, generating names, ViT |
| `p2ch10/` | Diffusion: forward diffusion, U-Net, DiT, training on MNIST (`utils.py`) |
| `p2ch11/` | No code (README only) |
| `p2ch12/` | LUNA lung CT dataset loading (`dsets.py`, `vis.py`) |
| `p2ch13/` | LUNA nodule classifier: `dsets.py`, `model.py`, `prepcache.py`, `training.py` |
| `p2ch14/` | LUNA classifier with balancing, augmentation, F1 metric (same file set as p2ch13) |
| `p2ch15/` | Segmentation with SAM: point prompts, CT slices, dataset creation, fine-tuning, inference |
| `p2ch16/` | Distributed training: torchrun, collectives, DDP, model/pipeline parallel, FSDP, device mesh, tensor parallel |
| `p2ch17/` | Deployment: `torch.compile`, `torch.export`, ONNX, quantization, profiler, ExecuTorch, FastAPI and Gradio servers, `c++/` (CMake example) |
| `p2ch12_explore_data.ipynb`, `p2ch14_explore_data.ipynb` | Data exploration notebooks for the LUNA chapters (3.4 MB and 1.1 MB) |
| `util/` | Shared helpers for part 2: `logconf.py`, `util.py` (`importstr`), `disk.py` (diskcache), `augmentation.py`, `unet.py` |
| `data/` | Small book data by chapter (`p1ch2` 120M, `p1ch4` 55M, `part2/luna` CSVs, `part2/models` pretrained `.state` files) |
| `ch2/`, `ch3/`, `ch4/` | Untracked. Leo's own chapter notebooks (pretrained networks, inpainting, tensors, image/CT/wine data) |
| `pyproject.toml`, `poetry.lock`, `requirements.txt` | Upstream dependency specs (torch not included) |
| `__init__.py` | Empty, makes the root importable |
| `.venv/` | Gitignored. This repo's own env |

## Run
- Interpreter: `/mnt/data0/shared/leot/dlwpt-code-2e/.venv/bin/python` (uv, Python 3.12 from `/mnt/data0/shared/leot/.uv-pythons`, torch 2.14.1+cu130, plus `requirements.txt` and ipykernel). Use it only inside this repo.
- Notebooks: Jupyter kernel "Python (dlwpt-code-2e)" (`~/.local/share/jupyter/kernels/dlwpt-code-2e/kernel.json`, points at the venv python).
- Part 2 apps use relative imports and `util.*`, so run them as modules from the repo root:
  - `.venv/bin/python -m p2ch13.prepcache --num-workers 4`
  - `.venv/bin/python -m p2ch13.training --epochs 1 --num-workers 4`
  - `.venv/bin/python -m p2ch14.training --epochs 2 --balanced balanced`
- Distributed examples: `.venv/bin/torchrun --nproc_per_node=2 fsdp_example.py` from `p2ch16/` (usage line in the script header).
- `ch2/3_inpainting.py` (Stable Diffusion 2 inpainting via diffusers) opens `../data/p1ch2/*.jpg`, so run it from `ch2/`.

## Outputs, jobs, logs
- No sbatch scripts in `/mnt/data0/shared/leot/Jobs/` and no `/mnt/data0/shared/leot/output/` folder for this repo.
- `p2ch13`/`p2ch14` training writes TensorBoard logs to `runs/<tb-prefix>/<time>-trn_cls-<comment>` relative to the cwd (`.gitignore` has `code/runs/*`).
- `util/disk.py` caches samples in `data-unversioned/cache/` (gitignored).
- `ch2/3_inpainting.py` saves `ch2/3_inpainting_out.png`.

## Dependencies outside this folder
- `/mnt/data0/shared/leot/.uv-pythons/cpython-3.12-linux-x86_64-gnu` (base interpreter of `.venv`).
- No code references to `/mnt/data0/shared` or `sys.path` edits.

## Conventions and gotchas
- AGENTS.md: this venv is for this repo only. Every other workspace repo uses `/mnt/data0/shared/leot/.venv`.
- The LUNA CT scans are not here. `p2ch12`/`p2ch13`/`p2ch14` `dsets.py` read `data-unversioned/part2/luna/subset*/*.mhd`, which does not exist. Only the annotation and candidate CSVs are in `data/part2/luna/`.
- `ch3/tensor.ipynb` overwrites the tracked `data/p1ch3/ourpoints.t` (it shows as modified in `git status`).
- `.gitignore` excludes `*.pt`, `*.pth`, `*.pt2`, `*.onnx`, `*.zip`, `*.json`, `notes.md`, `build/`, `libtorch/`, `.gradio`, `data-unversioned/`.
- `p2ch16/3_torchrun.py` header says `2_torchrun.py` (stale name in the usage comment).
