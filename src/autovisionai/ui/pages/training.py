"""Training page for AutoVisionAI Streamlit UI."""

import threading
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import requests
import streamlit as st

from autovisionai.ui.utils import add_sidebar_api_status


class TrainingProgressTracker:
    """Tracks training progress for display."""

    def __init__(self):
        self.is_training = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.train_loss = []
        self.val_loss = []
        self.learning_rate = []
        self.logs = []
        self.start_time = None

    def start_training(self, total_epochs: int):
        """Start training simulation."""
        self.is_training = True
        self.current_epoch = 0
        self.total_epochs = total_epochs
        self.train_loss = []
        self.val_loss = []
        self.learning_rate = []
        self.logs = []
        self.start_time = datetime.now(timezone.utc)

    def update_progress(self, epoch: int, train_loss: float, val_loss: float, lr: float, log_msg: str):
        """Update training progress."""
        self.current_epoch = epoch
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.learning_rate.append(lr)
        self.logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {log_msg}")

    def stop_training(self):
        """Stop training."""
        self.is_training = False


def simulate_training_progress(tracker: TrainingProgressTracker, config: dict):
    """Simulate training progress for demonstration."""
    import math
    import random

    total_epochs = config["epochs"]
    tracker.start_training(total_epochs)

    for epoch in range(1, total_epochs + 1):
        if not tracker.is_training:  # Allow stopping
            break

        # Simulate realistic loss curves
        base_train_loss = 0.8 * math.exp(-epoch / 20) + 0.1
        base_val_loss = base_train_loss + 0.05 + random.gauss(0, 0.02)

        # Add some noise
        train_loss = base_train_loss + random.gauss(0, 0.01)
        val_loss = base_val_loss + random.gauss(0, 0.015)

        # Learning rate schedule
        lr = config["learning_rate"] * (0.95 ** (epoch // 10))

        log_msg = f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {lr:.6f}"

        tracker.update_progress(epoch, train_loss, val_loss, lr, log_msg)

        # Simulate epoch time
        time.sleep(0.5)  # Half second per epoch for demo

    if tracker.is_training:  # Completed normally
        tracker.logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Training completed!")
        tracker.stop_training()


def plot_training_curves(tracker: TrainingProgressTracker):
    """Plot training and validation loss curves."""
    if not tracker.train_loss:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curves
    epochs = range(1, len(tracker.train_loss) + 1)
    ax1.plot(epochs, tracker.train_loss, label="Training Loss", color="blue")
    ax1.plot(epochs, tracker.val_loss, label="Validation Loss", color="red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Learning rate
    ax2.plot(epochs, tracker.learning_rate, label="Learning Rate", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale("log")

    plt.tight_layout()
    return fig


def submit_training_job(config: dict) -> bool:
    """Submit a training job to the API."""
    try:
        api_url = st.session_state.get("api_base_url", "http://localhost:8000")
        training_url = f"{api_url}/training/start"

        response = requests.post(training_url, json=config, timeout=10)
        response.raise_for_status()

        result = response.json()
        st.success(f"Training job submitted! Job ID: {result.get('job_id', 'N/A')}")
        return True

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to API at {api_url}. Make sure the API server is running.")
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    return False


# Initialize training tracker in session state
if "training_tracker" not in st.session_state:
    st.session_state.training_tracker = TrainingProgressTracker()

tracker = st.session_state.training_tracker

# Main page content
st.title("🎯 Training")
st.markdown("Configure and monitor training jobs for car segmentation models.")

add_sidebar_api_status()
# Training configuration form
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚙️ Training Configuration")

    with st.form("training_config_form"):
        # Model selection
        model_architecture = st.selectbox(
            "🤖 Model Architecture",
            ["unet", "fast_scnn", "mask_rcnn"],
            index=0,
            help="Choose the model architecture to train",
        )

        # Dataset configuration
        st.markdown("**📁 Dataset Configuration**")
        dataset_path = st.text_input(
            "Dataset Path",
            value="/data/car_segmentation",
            help="Path to the training dataset",
        )

        train_split = st.slider("Training Split", 0.1, 0.9, 0.8, 0.05, help="Fraction of data used for training")

        # Training hyperparameters
        st.markdown("**🔧 Training Hyperparameters**")
        col_lr, col_bs = st.columns(2)

        with col_lr:
            learning_rate = st.number_input(
                "Learning Rate",
                value=0.001,
                min_value=0.0001,
                max_value=0.1,
                step=0.0001,
                format="%.4f",
                help="Initial learning rate for training",
            )

        with col_bs:
            batch_size = st.selectbox("Batch Size", [4, 8, 16, 32, 64], index=2, help="Training batch size")

        col_ep, col_pat = st.columns(2)

        with col_ep:
            epochs = st.number_input(
                "Epochs", value=50, min_value=1, max_value=1000, step=1, help="Number of training epochs"
            )

        with col_pat:
            patience = st.number_input(
                "Early Stopping Patience",
                value=10,
                min_value=1,
                max_value=50,
                step=1,
                help="Epochs to wait before early stopping",
            )

        # Advanced options
        with st.expander("🔬 Advanced Options", expanded=False):
            optimizer = st.selectbox("Optimizer", ["adam", "sgd", "adamw"], index=0)

            weight_decay = st.number_input(
                "Weight Decay",
                value=0.0001,
                min_value=0.0,
                max_value=0.01,
                step=0.0001,
                format="%.4f",
            )

            use_scheduler = st.checkbox("Use Learning Rate Scheduler", value=True)

            if use_scheduler:
                scheduler_type = st.selectbox("Scheduler Type", ["step", "cosine", "exponential"], index=0)
                scheduler_step_size = st.number_input("Step Size", value=10, min_value=1, max_value=100, step=1)

        # Submit button
        submitted = st.form_submit_button("🚀 Start Training", type="primary", use_container_width=True)

        if submitted:
            # Prepare configuration
            training_config = {
                "model_architecture": model_architecture,
                "dataset_path": dataset_path,
                "train_split": train_split,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
                "use_scheduler": use_scheduler,
            }

            if use_scheduler:
                training_config.update({"scheduler_type": scheduler_type, "scheduler_step_size": scheduler_step_size})

            # Try to submit to API first
            if submit_training_job(training_config):
                st.info("Training job submitted to API. Monitor progress below.")
            else:
                st.warning("API submission failed. Starting local simulation instead.")

            # Start local simulation regardless
            if not tracker.is_training:
                training_thread = threading.Thread(
                    target=simulate_training_progress, args=(tracker, training_config), daemon=True
                )
                training_thread.start()

with col2:
    st.subheader("📊 Training Progress")

    if not tracker.is_training and tracker.current_epoch == 0:
        st.info("Configure training parameters and click 'Start Training' to begin.")
    else:
        # Progress metrics
        if tracker.is_training or tracker.current_epoch > 0:
            col_ep, col_total, col_time = st.columns(3)

            with col_ep:
                st.metric("Current Epoch", tracker.current_epoch)

            with col_total:
                st.metric("Total Epochs", tracker.total_epochs)

            with col_time:
                if tracker.start_time:
                    elapsed = datetime.now(timezone.utc) - tracker.start_time
                    st.metric("Elapsed Time", f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s")

            # Progress bar
            if tracker.total_epochs > 0:
                progress = tracker.current_epoch / tracker.total_epochs
                st.progress(progress, text=f"Epoch {tracker.current_epoch}/{tracker.total_epochs}")

            # Current losses
            if tracker.train_loss:
                col_train, col_val = st.columns(2)
                with col_train:
                    st.metric("Train Loss", f"{tracker.train_loss[-1]:.4f}")
                with col_val:
                    st.metric("Val Loss", f"{tracker.val_loss[-1]:.4f}")

        # Stop training button
        if tracker.is_training:
            if st.button("⏹️ Stop Training", type="secondary", use_container_width=True):
                tracker.stop_training()
                st.warning("Training stopped by user.")

        # Training curves
        if tracker.train_loss:
            st.markdown("**📈 Training Curves**")
            fig = plot_training_curves(tracker)
            if fig:
                st.pyplot(fig)
                plt.close(fig)

        # Training logs
        if tracker.logs:
            st.markdown("**📝 Training Logs**")
            log_container = st.container()
            with log_container:
                # Show last 10 logs
                for log in tracker.logs[-10:]:
                    st.text(log)

            # Download logs button
            if st.button("📄 Download Logs", use_container_width=True, key="download_logs_btn"):
                logs_text = "\n".join(tracker.logs)
                st.download_button(
                    label="📄 Download Training Logs",
                    data=logs_text,
                    file_name=f"training_logs_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    key="download_logs_file_btn",
                )

        # Auto-refresh for real-time updates
        if tracker.is_training:
            time.sleep(1)
            st.rerun()
