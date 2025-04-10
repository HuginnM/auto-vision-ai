# AutoVisionAI

**AutoVisionAI** is a computer vision project focused on **semantic segmentation of vehicles** in images. This repository serves as a demonstration of professional ML engineering practices, model development workflows, and production-ready architecture.

The initial goal is to build a fully functional **U-Net model from scratch** to segment cars from street images. Future iterations may explore additional model architectures, training strategies, and MLOps integrations.

---

## 🚀 Project Goals
- Develop a high-performance vehicle segmentation model
- Showcase clean project architecture and engineering discipline
- Integrate **MLflow** and **Weights & Biases** for tracking and observability
- Build a testable and extensible ML pipeline with CI/CD

---

## 📁 Project Structure (src layout)

```bash
AutoVisionAI/
├── src/                      # Source code
│   └── autovisionai/        # Main Python package
│       ├── models/          # Model architectures (e.g., UNet, Fast-SCNN)
│       ├── processing/      # Transforms, datasets, datamodules
│       ├── utils/           # Helper functions, metrics, visualizations
│       └── __init__.py
│
├── configs/                 # YAML config files (paths, hyperparams)
├── notebooks/               # Optional exploration, visualization
├── experiments/             # Model checkpoints, logs (MLflow, W&B)
├── tests/                   # Pytest unit & integration tests
├── train.py                 # Training entrypoint
├── inference.py             # Inference script
├── pyproject.toml           # Build configuration
├── requirements.txt         # Pip requirements
├── .gitignore               # Git exclusions
└── README.md
```

---

## 🧪 Testing

Tests are located under the `tests/` directory and run automatically in CI. To run locally:

```bash
pytest tests/
```

---

## 🧠 Git Workflow (Trunk-based)

- `main` — stable, production-ready code only
- `feature/*` — development branches (e.g., `feature/unet`, `feature/refactor-dataloader`)
- All contributions go through **Pull Requests** with CI validation

---

## 🛠️ Planned Integrations
- ✅ PyTorch Lightning
- ✅ U-Net model and others
- 🔜 MLflow tracking
- 🔜 Weights & Biases
- 🔜 Model registry + inference API

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute it for personal or commercial purposes, as long as the original copyright notice is included.

---

## 🧑‍💻 Author
Developed by Arthur Sobol — ML/AI Engineer.
