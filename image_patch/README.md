## Image patch experiments (EvenCodeNet / IPU)

This folder contains the code used in Section 6 of 
“Efficient Representation of Natural Image Patches” (see the top‑level `README.md` for the paper link).

The goal here is to learn a sparse binary population code for small natural image patches.

### Main files

- `config.yaml` – experiment configuration (data path, patch size, model, optimizer, training schedule, checkpoint naming).
- `config.py` – small helper to load/save `config.yaml` into a `Config` object.
- `util.py` – model and data utilities:
  - `EvenCodeNet` and custom layers (`MLPConv2d`, `ParallelMLPConv2d`, etc.).
  - Datasets for images and videos (`ImageFolder`, `ImagePatchesWithLabel`, `VideoPatches`).
  - `save_model` / `load_model` helpers and parameter counting.
- `train_image_encoder.py` – trains EvenCodeNet on natural images using the settings in `config.yaml`.
- `gen_results.py` – loads a trained checkpoint and generates diagnostics and visualizations.
- `plot_utils.py` – plotting helpers (loss curves, histograms, receptive fields, PCA feature‑map visualizations, similar‑patch mosaics).
- `log_utils.py` – logger setup; writes logs such as `train.log` and `util.log`.
- `test_images/` – example images used by `gen_results.py` for feature‑map visualizations.
- `study_feature_map.ipynb` – Jupyter notebook for interactively exploring and visualizing feature maps and activation patterns of a trained encoder.

### Setup

1. Create a Python environment (Python 3.9+ recommended).
2. Install dependencies from the project root:
   ```bash
   cd ..  # go to repo root
   pip install -r requirement.txt
   ```
3. Edit `image_patch/config.yaml`:
   - Set `image_path` to a folder of natural images on your machine (e.g., COCO, ImageNet, or any large image collection).
   - Set `device` to `"cuda:0"` (GPU) or `"cpu"` depending on your hardware.
   - Optionally adjust patch size (`input_height`, `input_width`), batch sizes, optimizer, and number of epochs.

### Training the encoder

From the repo root:

```bash
cd image_patch
python train_image_encoder.py
```

This will:

- Build EvenCodeNet from `config.yaml`.
- Sample patches from the images at `image_path` using `ImagePatchesWithLabel`.
- Train the last layer(s) with the even code loss.
- Periodically save checkpoints as `checkpoint_epoch_{epoch}.pt` and append to `train.log` and `util.log`.

To resume from a checkpoint, set `init_from_epoch` in `config.yaml` to the epoch index you want to start from and re‑run `train_image_encoder.py`.

### Generating plots and analyses

Once training has finished (or you have at least one checkpoint), run:

```bash
cd image_patch
python gen_results.py
```

This will:

- Load the last epoch’s checkpoint (by default `checkpoint_epoch_{n_epoch-1}.pt`).
- Save a loss curve, histograms of activations, distributions over active nodes, and other summary plots (e.g. `loss.png`, `outputs_hist.png`, `outputs_dist.png`, `num_samples_vs_num_activenodes.png`, `receptive_fields.png`, `similar_patches_seed*.png`) into `plots/`.
- Save sampled patches to `test_patches.npy` for later analysis.
- If `as_gray` is `False`, generate a PCA visualization of feature maps for a sample image in `test_images/` (written as `feature_maps_pca.png` in `plots/`).
- Create mosaics of similar patches and visualizations of “receptive fields” implied by the learned binary patterns.

For interactive inspection of feature maps on individual images, you can also open the notebook:

```bash
cd image_patch
jupyter notebook study_feature_map.ipynb
```

The notebook loads `config.yaml` and the last checkpoint (via `fname_checkpoint` and `n_epoch`), runs the encoder on a sample image from `test_images/` (see the `filename` variable in the notebook), and displays the input and all output channels as a grid so you can visually inspect the learned feature maps.

### Notes

- All plotting functions write to `plots/`, which is created automatically.
- You can tweak model depth, number of channels, or patch size by editing the `layers` section in `config.yaml`.
