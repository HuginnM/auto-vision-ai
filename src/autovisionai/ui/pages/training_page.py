"""Training page for AutoVisionAI Streamlit UI."""

import random
import threading
import time
from datetime import datetime, timezone
from typing import Dict

import matplotlib.pyplot as plt
import requests
import streamlit as st


def show_training_page():
    """Show the training page with hyperparameters and training controls."""
    st.title("Training")
    st.markdown("Configure training parameters and monitor training progress in real-time.")

    # Initialize training state
    if "training_data" not in st.session_state:
        st.session_state.training_data = {
            "is_training": False,
            "experiment_name": "",
            "progress": {},
            "loss_history": [],
            "epochs_history": [],
            "training_logs": [],
        }

    # Main content layout
    col1, col2 = st.columns([1, 1])

    with col1:
        show_training_config()

    with col2:
        show_training_progress()


def show_training_config():
    """Show training configuration form."""
    st.subheader("Training Configuration")

    with st.form("training_config"):
        # Experiment configuration
        st.markdown("#### Experiment Settings")
        experiment_name = st.text_input(
            "Experiment Name",
            value=f"experiment_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            help="Unique name for this training experiment",
            key="experiment_name_input",
        )

        model_name = st.selectbox(
            "Model Architecture",
            ["unet", "fast_scnn", "mask_rcnn"],
            index=0,
            help="Select the model architecture to train",
            key="training_model_select",
        )

        # Model information
        model_info = get_model_info(model_name)
        st.info(f"**{model_info['name']}**: {model_info['description']}")

        # Hyperparameters
        st.markdown("#### Hyperparameters")

        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input(
                "Batch Size", min_value=1, max_value=64, value=4, help="Number of samples per training batch"
            )

            epoch_patience = st.number_input(
                "Early Stopping Patience",
                min_value=1,
                max_value=20,
                value=2,
                help="Number of epochs to wait before early stopping",
            )

        with col2:
            max_epochs = st.number_input(
                "Max Epochs", min_value=1, max_value=1000, value=100, help="Maximum number of training epochs"
            )

        # Data augmentation
        st.markdown("#### Data Augmentation")

        col1, col2, col3 = st.columns(3)
        with col1:
            use_resize = st.checkbox("Use Resize", value=False, help="Apply resize augmentation")
        with col2:
            use_random_crop = st.checkbox("Use Random Crop", value=False, help="Apply random crop augmentation")
        with col3:
            use_hflip = st.checkbox("Use Horizontal Flip", value=False, help="Apply horizontal flip augmentation")

        # Submit button
        submit_button = st.form_submit_button(
            "Start Training",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.training_data["is_training"],
        )

        if submit_button:
            start_training(
                {
                    "experiment_name": experiment_name,
                    "model_name": model_name,
                    "batch_size": batch_size,
                    "epoch_patience": epoch_patience,
                    "use_resize": use_resize,
                    "use_random_crop": use_random_crop,
                    "use_hflip": use_hflip,
                    "max_epochs": max_epochs,
                }
            )

    # Stop training button
    if st.session_state.training_data["is_training"]:
        if st.button("Stop Training", type="secondary", use_container_width=True, key="stop_training_btn"):
            stop_training()

    # Training history
    if st.session_state.training_data.get("experiment_name"):
        with st.expander("Training Configuration Summary", expanded=False):
            st.write(f"**Experiment:** {st.session_state.training_data['experiment_name']}")
            status_text = "Running" if st.session_state.training_data["is_training"] else "Stopped"
            st.write(f"**Status:** {status_text}")


def show_training_progress():
    """Show training progress and metrics."""
    st.subheader("Training Progress")

    if not st.session_state.training_data["is_training"] and not st.session_state.training_data["progress"]:
        st.info("Configure training parameters and click 'Start Training' to begin.")
        return

    # Training status
    if st.session_state.training_data["is_training"]:
        st.success("Training in progress...")
        # Auto-refresh every 2 seconds during training
        time.sleep(2)
        st.rerun()
    else:
        if st.session_state.training_data["progress"]:
            status = st.session_state.training_data["progress"].get("status", "unknown")
            if status == "completed":
                st.success("Training completed!")
            elif status == "error":
                st.error("Training failed!")
            else:
                st.warning(f"Training status: {status}")

    # Progress metrics
    if st.session_state.training_data["progress"]:
        progress = st.session_state.training_data["progress"]

        # Progress bar
        if progress.get("total_epochs", 0) > 0:
            current_epoch = progress.get("current_epoch", 0)
            total_epochs = progress.get("total_epochs", 1)
            progress_percent = current_epoch / total_epochs
            st.progress(progress_percent, text=f"Epoch {current_epoch}/{total_epochs}")

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Epoch", progress.get("current_epoch", 0))
        with col2:
            current_loss = progress.get("current_loss", float("inf"))
            if current_loss != float("inf"):
                st.metric("Current Loss", f"{current_loss:.6f}")
            else:
                st.metric("Current Loss", "N/A")
        with col3:
            best_loss = progress.get("best_loss", float("inf"))
            if best_loss != float("inf"):
                st.metric("Best Loss", f"{best_loss:.6f}")
            else:
                st.metric("Best Loss", "N/A")

    # Loss graph
    show_loss_graph()

    # Training logs
    show_training_logs()


def show_loss_graph():
    """Show loss progression graph."""
    if not st.session_state.training_data["loss_history"]:
        return

    st.markdown("#### Loss Progression")

    epochs = st.session_state.training_data["epochs_history"]
    losses = st.session_state.training_data["loss_history"]

    if epochs and losses:
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, losses, "b-", linewidth=2, label="Training Loss", marker="o", markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss Over Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set reasonable y-axis limits
        if losses:
            min_loss = min(losses)
            max_loss = max(losses)
            if min_loss != max_loss:
                margin = (max_loss - min_loss) * 0.1
                ax.set_ylim(min_loss - margin, max_loss + margin)

        # Display the plot
        st.pyplot(fig)
        plt.close()

        # Add some statistics
        if len(losses) > 1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Starting Loss", f"{losses[0]:.6f}")
            with col2:
                st.metric("Current Loss", f"{losses[-1]:.6f}")
            with col3:
                improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
                st.metric("Improvement", f"{improvement:.1f}%")


def show_training_logs():
    """Show training logs."""
    if not st.session_state.training_data["training_logs"]:
        return

    st.markdown("#### Training Logs")

    with st.expander("View Recent Logs", expanded=False):
        logs_container = st.container()
        with logs_container:
            # Show last 20 logs
            recent_logs = st.session_state.training_data["training_logs"][-20:]
            for log in recent_logs:
                st.text(log)

    # Clear logs button
    if st.button("Clear Logs", key="clear_logs_btn"):
        st.session_state.training_data["training_logs"] = []
        st.success("Logs cleared!")


def start_training(config: Dict):
    """Start training with the given configuration."""
    try:
        # Reset training data
        st.session_state.training_data = {
            "is_training": True,
            "experiment_name": config["experiment_name"],
            "progress": {},
            "loss_history": [],
            "epochs_history": [],
            "training_logs": [f"Starting training with {config['model_name']} model..."],
        }

        # Send training request
        api_url = st.session_state.api_base_url
        training_url = f"{api_url}/train/"

        response = requests.post(training_url, json=config, timeout=10)
        response.raise_for_status()

        result = response.json()
        st.success(f"Training started! Status: {result.get('status', 'unknown')}")

        # Start progress simulation (in real implementation, use WebSocket)
        simulate_training_progress()

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to API at {api_url}. Make sure the API server is running.")
        st.session_state.training_data["is_training"] = False
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to start training: {e}")
        st.session_state.training_data["is_training"] = False
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.training_data["is_training"] = False


def simulate_training_progress():
    """Simulate training progress for demonstration purposes."""

    def update_progress():
        max_epochs = 10
        for epoch in range(1, max_epochs + 1):
            if not st.session_state.training_data["is_training"]:
                break

            # Simulate decreasing loss with some noise
            base_loss = 1.0 / (epoch * 0.5 + 1)
            noise = random.uniform(-0.1, 0.1)
            current_loss = max(0.001, base_loss + noise)

            # Update best loss
            previous_best = st.session_state.training_data.get("best_loss", float("inf"))
            if "progress" in st.session_state.training_data:
                previous_best = st.session_state.training_data["progress"].get("best_loss", float("inf"))
            best_loss = min(previous_best, current_loss)

            progress_data = {
                "current_epoch": epoch,
                "total_epochs": max_epochs,
                "current_loss": current_loss,
                "best_loss": best_loss,
                "status": "training",
                "detail": f"Training epoch {epoch}/{max_epochs}",
                "output_logs": [f"Epoch {epoch}: loss = {current_loss:.6f}"],
            }

            # Update session state
            st.session_state.training_data["progress"] = progress_data
            st.session_state.training_data["epochs_history"].append(epoch)
            st.session_state.training_data["loss_history"].append(current_loss)
            st.session_state.training_data["training_logs"].append(f"Epoch {epoch}: loss = {current_loss:.6f}")

            time.sleep(3)  # Simulate training time

        # Mark as completed
        if st.session_state.training_data["is_training"]:
            st.session_state.training_data["progress"]["status"] = "completed"
            st.session_state.training_data["progress"]["detail"] = "Training completed successfully!"
            st.session_state.training_data["training_logs"].append("Training completed!")
            st.session_state.training_data["is_training"] = False

    # Start simulation in background
    thread = threading.Thread(target=update_progress)
    thread.daemon = True
    thread.start()


def stop_training():
    """Stop the current training."""
    st.session_state.training_data["is_training"] = False
    if st.session_state.training_data["progress"]:
        st.session_state.training_data["progress"]["status"] = "stopped"
        st.session_state.training_data["progress"]["detail"] = "Training stopped by user"
    st.session_state.training_data["training_logs"].append("Training stopped by user.")
    st.warning("Training stopped by user.")


def get_model_info(model_name: str) -> dict:
    """Get information about a model."""
    model_info = {
        "unet": {
            "name": "U-Net",
            "description": "Classic encoder-decoder architecture for semantic segmentation",
            "params": "~31M",
            "speed": "Medium",
        },
        "fast_scnn": {
            "name": "Fast-SCNN",
            "description": "Fast Segmentation CNN optimized for real-time inference",
            "params": "~1.1M",
            "speed": "Fast",
        },
        "mask_rcnn": {
            "name": "Mask R-CNN",
            "description": "Instance segmentation with object detection capabilities",
            "params": "~44M",
            "speed": "Slow",
        },
    }
    return model_info.get(
        model_name, {"name": model_name, "description": "Unknown model", "params": "Unknown", "speed": "Unknown"}
    )
