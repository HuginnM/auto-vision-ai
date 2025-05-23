# AutoVisionAI

AutoVisionAI is a modular computer vision pipeline focused on semantic segmentation of vehicles. It demonstrates industry-grade engineering practices, reproducible experimentation, and scalable ML model development workflows.

## Key Features

- Modular architecture built using PyTorch Lightning and TorchVision
- U-Net implementation from scratch
- Fast-SCNN implementation from scratch
- Mask R-CNN with pretrained TorchVision backbone and custom prediction heads
- Custom LightningModule training wrappers with configurable transforms
- Unit and integration tests via pytest and automated CI/CD via GitHub Actions
- Strict `src/` layout for scalable package management
- Modern Python tooling: `uv` for dependency management, `ruff` for linting & formatting
- Training visualization using TensorBoard (W&B & MLflow support planned)
- Trunk-based Git development workflow

## Project Layout

```
AutoVisionAI/
├── .github/                 # GitHub Actions CI config
├── data/                   # Images and segmentation masks
├── experiments/            # Logs and model checkpoints
├── src/
│   └── autovisionai/
│       ├── configs/        # Confuse/YAML configuration
│       ├── models/         # UNet, FastSCNN, MaskRCNN + Lightning trainers
│       ├── processing/     # Dataset and DataModules
│       ├── utils/          # Metrics, helpers, visualization
│       └── train.py        # Training script
├── tests/                  # Unit + integration tests
│   ├── models/
│   ├── processing/
│   ├── utils/
│   └── test_data/
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── pyproject.toml
├── LICENSE
├── README.md
└── uv.lock
```

## Installation:

### With uv (recommended)
```bash
uv pip install -e ".[all]"
uv run pre-commit install
```

### With pip (legacy)
```bash
pip install -e ".[all]"
pre-commit install
```
> `.[all]` includes pytest, ruff, pre-commit, and other development tools.<br>
> Omit `[all]` if you only want to install the core package for production use without development dependencies.

---
> If you don't have `uv` installed, you can get it with:

```bash
curl -Ls https://astral.sh/uv/install.sh | bash
```

## Supported Models

### 1. U-Net (from scratch)

- Semantic segmentation model
- Designed for small-scale experiments
- No pretrained weights
- Paper: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

### 2. Fast-SCNN (from scratch)

- Real-time semantic segmentation
- Fully implemented from scratch
- Lightweight and mobile-friendly
- Paper: [Fast-SCNN: Fast Semantic Segmentation Network](https://arxiv.org/abs/1902.04502)

### 3. Mask R-CNN (TorchVision pretrained)

- Instance segmentation using pretrained ResNet50-FPN backbone
- `box_predictor` and `mask_predictor` layers replaced and fine-tuned
- Integrated with PyTorch Lightning `Trainer`
- Paper: [Mask R-CNN](https://arxiv.org/abs/1703.06870)

## Qualitative Results

### Semantic Segmentation

The U-Net and Fast-SCNN models were implemented from scratch and trained on a limited synthetic dataset (Carvana). Due to resource constraints and intentionally lightweight training, they perform well on the in-domain data but may generalize poorly to out-of-distribution samples.

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td width="220">
      <strong>U-Net</strong><br>
      Semantic segmentation model trained and built from scratch on a small dataset.
    </td>
    <td>
      <img src="https://drive.google.com/uc?id=1DWXLDXvaR_XMH50uq1qamsT1WdV9IzQZ" width="100%" alt="U-Net Prediction">
    </td>
  </tr>
  <tr>
    <td width="220">
      <strong>Fast-SCNN</strong><br>
      Real-time lightweight segmentation model built from scratch. Fast and efficient for edge devices.
    </td>
    <td>
      <img src="https://drive.google.com/uc?id=1F13Q6LLTcYc2CIF0mhlqIbx6PC0kAyQX" width="100%" alt="Fast-SCNN Prediction">
    </td>
  </tr>
</table>

> These results illustrate solid segmentation performance within the training domain, achieved using entirely custom implementations without pretrained weights.

### Instance Segmentation

The Mask R-CNN model utilizes a pretrained ResNet50-FPN backbone from TorchVision, with the `box_predictor` and `mask_predictor` heads replaced and fine-tuned on a small dataset. Despite limited data, the model demonstrates good generalization, producing correct instance masks even on real-world images found online.

<table style="width:100%; table-layout:fixed;">
  <tr>
    <td width="220">
      <strong>Mask R-CNN</strong><br>
      Instance segmentation using a pretrained backbone with fine-tuned predictors. <br>(For one uncommon car)
    </td>
    <td>
      <img src="https://drive.google.com/uc?id=1uQM6gWPbQHCccwYoxBd_bdjOHzIPjBpX" width="100%" alt="Mask R-CNN Prediction">
    </td>
  </tr>
  <tr>
    <td width="220">
      <strong>Mask R-CNN</strong><br>
      <br>(For two instances of very rare cars)
    </td>
    <td>
      <img src="https://drive.google.com/uc?id=1om1YSYS6k3q3ZTjkJ3k-e_BZZyUs7iAo" width="100%" alt="Mask R-CNN Prediction">
    </td>
  </tr>
</table>

> The model demonstrates robust instance segmentation capabilities despite a tiny dataset.

## Training

```python
from autovisionai.train import train_model
from autovisionai.models.mask_rcnn.mask_rcnn_trainer import MaskRCNNTrainer

model = MaskRCNNTrainer()
train_model(
    exp_number=3,
    model=model,
    batch_size=4,
    max_epochs=1,
    use_resize=False,
    use_random_crop=True,
    use_hflip=True
)
```

## Inference

```python
from autovisionai.utils.utils import get_input_image_for_inference
from autovisionai.models.mask_rcnn.mask_rcnn_inference import model_inference
from autovisionai.utils.utils import show_pic_and_pred_instance_masks
from autovisionai.configs.config import CONFIG

exp_n = 1
image = get_input_image_for_inference(url="https://your_image.com/img.jpg")
model_path = Path(CONFIG["trainer"]["logs_and_weights_root"].get(confuse.Filename())) / f"exp_{exp_n}/weights/model.pt"
_, _, scores, masks = model_inference(model_path, image)
show_pic_and_pred_instance_masks(image, masks, scores)
```

## CI/CD and DevOps

- ✅ Continuous Integration via **GitHub Actions**
- ✅ Full `pytest` test coverage (executed in CI)
- ✅ TensorBoard logging for experiment tracking
- ✅ Migration to `uv` + `ruff` for dependency + linting
- 🔜 Continuous Deployment workflows
- 🔜 Weights & Biases logging support
- 🔜 MLflow tracking and model registry
- 🔜 Web demo with Docker + Kubernetes packaging

## 🛠 Modern Python Tooling

AutoVisionAI is fully equipped with modern Python development tooling:

- **uv** — ultra-fast dependency resolver and `pip`/`venv` replacement
- **ruff** — unified linter and formatter (up to 100× faster than flake8/black)
- **pre-commit** — enforces code quality before every commit

### Configured Hooks:
- `ruff`: linter (`E`, `F`, `B`, `I`, etc.)
- `ruff-format`: formatter (black-compatible)
- `check-yaml`: ensures valid `.yaml` files
- `end-of-file-fixer`: enforces trailing newlines
- `trailing-whitespace`: removes trailing spaces

All configurations are centralized in `pyproject.toml`.
The project uses `[project.optional-dependencies]` for maximum compatibility with `pip`, `tox`, and `Poetry`.
Experimental `[dependency-groups]` are pre-configured and will be activated once supported by `pip`.

> For installation instructions, including `uv` and `pre-commit` setup, see [Installation](#installation).

## Git Workflow

- `main`: stable production-ready code
- `feature/*`: short-lived branches for isolated features
- PRs are required for merging; trunk-based development style

## License

MIT License. See [LICENSE](LICENSE) file.

## Author

Developed and maintained by Arthur Sobol
