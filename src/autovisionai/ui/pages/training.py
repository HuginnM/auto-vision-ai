"""Training page for AutoVisionAI Streamlit UI."""

import asyncio
import json
import threading
import time
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import requests
import streamlit as st
import websockets

from autovisionai.core.configs import CONFIG
from autovisionai.core.utils.common import get_run_name
from autovisionai.ui.utils import configure_sidebar, format_model_name


class TrainingProgressTracker:
    """Tracks training progress for display."""

    def __init__(self):
        self.is_training = False
        self.current_epoch = -1
        self.total_epochs = 0
        self.train_loss = []
        self.val_loss = []
        self.learning_rate = []
        self.logs = []
        self.start_time = None
        self.websocket = None
        self.ws_thread = None
        # Batch tracking per epoch
        self.current_batch = -1
        self.total_batches = 0

    def start_training(self, total_epochs: int, batch_size: int):
        """Start training simulation."""
        self.is_training = True
        # self.current_epoch = -1
        self.total_epochs = total_epochs
        self.train_loss = []
        self.val_loss = []
        self.learning_rate = []
        self.logs = []
        self.start_time = datetime.now(timezone.utc)
        # self.current_batch = -1
        self.total_batches = batch_size

    def update_progress(
        self,
        epoch: int,
        current_batch: int,
        total_batches: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        log_msg: str,
    ):
        """Update training progress."""
        self.current_epoch = epoch
        self.current_batch = current_batch
        self.total_batches = total_batches
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.learning_rate.append(lr)
        self.logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {log_msg}")

    def stop_training(self):
        """Stop training."""
        self.is_training = False
        if self.websocket:
            asyncio.run(self.websocket.close())
            self.websocket = None
        if self.ws_thread:
            self.ws_thread.join()
            self.ws_thread = None


def simulate_training_progress(tracker: TrainingProgressTracker, config: dict):
    """Simulate training progress for demonstration in case if the real training doesn't work for some reason."""
    import math
    import random

    total_epochs = config["max_epochs"]
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

        tracker.update_progress(
            epoch=epoch,
            current_batch=0,
            total_batches=config.get("batch_size", 0),
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr,
            log_msg=log_msg,
        )

        # Simulate epoch time
        time.sleep(1)  # Increased to 1 second per epoch for better visibility

    if tracker.is_training:  # Completed normally
        tracker.logs.append(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Training completed!")
        tracker.stop_training()


def plot_training_curves(tracker: TrainingProgressTracker):
    """Plot training and validation loss curves."""
    if not tracker.train_loss:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Training loss
    train_epochs = range(1, len(tracker.train_loss) + 1)
    ax1.plot(train_epochs, tracker.train_loss, label="Training Loss", color="blue")

    # Validation loss (may have fewer points)
    if tracker.val_loss:
        val_epochs = range(1, len(tracker.val_loss) + 1)
        ax1.plot(val_epochs, tracker.val_loss, label="Validation Loss", color="red")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # Learning rate (ensure equal length to values being plotted)
    if tracker.learning_rate:
        lr_epochs = range(1, len(tracker.learning_rate) + 1)
        ax2.plot(lr_epochs, tracker.learning_rate, label="Learning Rate", color="green")

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale("log")

    plt.tight_layout()
    return fig


async def monitor_training_websocket(tracker: TrainingProgressTracker, experiment_name: str):
    """Monitor training progress via WebSocket."""
    api_url = st.session_state.get("api_base_url", "http://localhost:8000")
    ws_url = f"ws://{api_url.split('://')[1]}/train/ws/{experiment_name}"

    try:
        async with websockets.connect(ws_url) as websocket:
            tracker.websocket = websocket
            st.info(f"Connected to training monitor for experiment: {experiment_name}")

            while True:
                try:
                    data = await websocket.recv()
                    progress = json.loads(data)

                    tracker.current_epoch = progress["current_epoch"]
                    tracker.total_epochs = progress["total_epochs"]
                    tracker.current_batch = progress["current_batch"]
                    tracker.total_batches = progress["total_batches"]
                    tracker.train_loss.append(progress["current_loss"])
                    tracker.val_loss.append(progress["val_loss"])
                    tracker.learning_rate.append(progress["learning_rate"])
                    tracker.logs.extend(progress["output_logs"])

                    # Check for terminal status
                    if progress.get("status") in ["completed", "error"]:
                        st.info(f"Training {progress.get('status')}")
                        tracker.stop_training()
                        break

                except websockets.ConnectionClosed:
                    st.warning("WebSocket connection closed by server.")
                    break

    except Exception as e:
        st.error(f"Error connecting to WebSocket: {e}")
        tracker.stop_training()


def start_websocket_monitoring(tracker: TrainingProgressTracker, experiment_name: str):
    """Start WebSocket monitoring in a separate thread."""

    def run_websocket():
        asyncio.run(monitor_training_websocket(tracker, experiment_name))

    tracker.ws_thread = threading.Thread(target=run_websocket, daemon=True)
    tracker.ws_thread.start()


def submit_training_job(config: dict) -> bool:
    """Submit a training job to the API."""
    try:
        api_url = st.session_state.get("api_base_url", "http://localhost:8000")
        training_url = f"{api_url}/train"

        # Use a shorter timeout for the initial request
        response = requests.post(training_url, json=config, timeout=5)
        response.raise_for_status()

        result = response.json()
        if result.get("status") == "success":
            st.success(f"Training job submitted! Experiment: {config['experiment_name']}")
            tracker.start_training(config["max_epochs"], config["batch_size"])
            return True
        else:
            st.error(f"Training submission failed: {result.get('detail', 'Unknown error')}")
            return False

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

configure_sidebar()

# Training configuration form
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("⚙️ Training Configuration")

    with st.form("training_config_form"):
        # Add experiment name field with standard name from get_experiment_name
        experiment_name = st.text_input(
            "Experiment Name",
            value=get_run_name("Experiment"),
            help="Name for this training experiment (auto-generated by default)",
        )
        # Model selection
        model_architecture = st.selectbox(
            "🤖 Model Architecture",
            CONFIG.models.available,
            index=0,
            help="Choose the model architecture to train",
            format_func=format_model_name,
        )

        # Get default values from CONFIG
        model_config = getattr(CONFIG.models, model_architecture)

        # Dataset configuration
        st.markdown("**📁 Dataset Configuration**")
        dataset_path = st.text_input(
            "Dataset Path",
            value=CONFIG.dataset.data_root,
            help="Path to the training dataset",
        )

        train_split = st.slider(
            "Training Split",
            0.1,
            0.9,
            CONFIG.datamodule.training_set_size,
            0.05,
            help="Fraction of data used for training",
        )

        # Training hyperparameters
        st.markdown("**🔧 Training Hyperparameters**")
        col_lr, col_bs = st.columns(2)

        with col_lr:
            learning_rate = st.number_input(
                "Learning Rate",
                value=model_config.optimizer.initial_lr,
                min_value=0.0001,
                max_value=0.1,
                step=0.0001,
                format="%.4f",
                help="Initial learning rate for training",
            )

        with col_bs:
            batch_size = st.selectbox(
                "Batch Size",
                [4, 8, 16, 32, 64],
                index=0,
                help="Training batch size",
            )

        col_ep, col_pat = st.columns(2)

        with col_ep:
            epochs = st.number_input(
                "Epochs",
                value=CONFIG.trainer.max_epoch,
                min_value=1,
                max_value=1000,
                step=1,
                help="Number of training epochs",
            )

        with col_pat:
            patience = st.number_input(
                "Early Stopping Patience",
                value=2,
                min_value=1,
                max_value=10,
                step=1,
                help="Epochs to wait before early stopping",
            )

        # Advanced options
        with st.expander("🔬 Advanced Options", expanded=False):
            optimizer = st.selectbox(
                "Optimizer",
                ["adam", "sgd", "adamw"],
                index=0,
                help="Optimizer to use for training",
            )

            weight_decay = st.number_input(
                "Weight Decay",
                value=model_config.optimizer.weight_decay,
                min_value=0.0,
                max_value=0.01,
                step=0.0001,
                format="%.4f",
                help="Weight decay for regularization",
            )

            use_scheduler = st.checkbox("Use Learning Rate Scheduler", value=True)

            if use_scheduler:
                scheduler_type = st.selectbox(
                    "Scheduler Type",
                    ["step", "cosine", "exponential"],
                    index=0,
                    help="Type of learning rate scheduler",
                )
                scheduler_step_size = st.number_input(
                    "Step Size",
                    value=model_config.lr_scheduler.step_size,
                    min_value=1,
                    max_value=100,
                    step=1,
                    help="Step size for learning rate scheduler",
                )
                scheduler_gamma = st.number_input(
                    "Gamma",
                    value=model_config.lr_scheduler.gamma,
                    min_value=0.1,
                    max_value=1.0,
                    step=0.1,
                    help="Gamma for learning rate scheduler",
                )

        # Submit button
        submitted = st.form_submit_button("🚀 Start Training", type="primary", use_container_width=True)

        if submitted:
            # Prepare configuration
            training_config = {
                "experiment_name": experiment_name,
                "model_name": model_architecture,
                "batch_size": batch_size,
                "epoch_patience": patience,
                "use_resize": False,
                "use_random_crop": False,
                "use_hflip": False,
                "max_epochs": epochs,
                "learning_rate": learning_rate,
                "optimizer": optimizer,
                "weight_decay": weight_decay,
            }

            if use_scheduler:
                training_config.update(
                    {
                        "scheduler_type": scheduler_type,
                        "scheduler_step_size": scheduler_step_size,
                        "scheduler_gamma": scheduler_gamma,
                    }
                )

            # Try to submit to API first
            if submit_training_job(training_config):
                st.info("Training job submitted to API. Monitor progress below.")
                # Start WebSocket monitoring
                start_websocket_monitoring(tracker, experiment_name)
            else:
                st.warning("API submission failed. Starting local simulation instead.")
                # Start local simulation
                if not tracker.is_training:
                    st.info("🚀 Starting training simulation...")
                    training_thread = threading.Thread(
                        target=simulate_training_progress, args=(tracker, training_config), daemon=True
                    )
                    training_thread.start()
                    st.rerun()  # Force immediate refresh

with col2:
    st.subheader("📊 Training Progress")

    # Add debug info to see current state
    if st.checkbox("Show Debug Info", key="debug_training"):
        st.write(f"Is Training: {tracker.is_training}")
        st.write(f"Current Epoch: {tracker.current_epoch}")
        st.write(f"Total Epochs: {tracker.total_epochs}")
        st.write(f"Current Batch: {tracker.current_batch}")
        st.write(f"Total Batches: {tracker.total_batches}")
        st.write(f"Number of loss entries: {len(tracker.train_loss)}")

    if not tracker.is_training and tracker.current_epoch == -1:
        st.info("Configure training parameters and click 'Start Training' to begin.")
    elif tracker.is_training and tracker.current_epoch == -1:
        st.warning("🔄 Initializing training... Please wait a moment.")
    else:
        # Progress metrics
        if tracker.is_training or tracker.current_epoch >= 0:
            col_ep, col_total, col_time = st.columns(3)

            with col_ep:
                st.metric("Current Epoch", tracker.current_epoch + 1)

            with col_total:
                st.metric("Total Epochs", tracker.total_epochs)

            with col_time:
                if tracker.start_time:
                    elapsed = datetime.now(timezone.utc) - tracker.start_time
                    st.metric("Elapsed Time", f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s")

            # Progress bar for overall epochs
            if tracker.total_epochs >= 0:
                progress = tracker.current_epoch / tracker.total_epochs
                st.progress(progress, text=f"Epoch {tracker.current_epoch + 1}/{tracker.total_epochs}")

            # Progress bar for batches within current epoch
            if tracker.total_batches >= 0 and tracker.is_training:
                batch_progress = tracker.current_batch / tracker.total_batches
                st.progress(
                    batch_progress,
                    text=f"Batch {tracker.current_batch}/{tracker.total_batches} in Epoch {tracker.current_epoch}",
                )

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
    time.sleep(0.5)
    st.rerun()
