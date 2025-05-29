import json
import threading
import time

import requests
import websocket


def test_train_endpoint(base_url="http://localhost:8000"):
    """
    Test the train endpoint with a sample training request.

    Args:
        base_url (str): The base URL of your API server (default: http://localhost:8000)

    Returns:
        dict: The response from the training endpoint
    """
    # Create a sample training request
    training_request = {
        "experiment_name": "test_experiment",
        "model_name": "unet",  # Using UNet as an example
        "batch_size": 4,
        "epoch_patience": 2,
        "use_resize": False,
        "use_random_crop": False,
        "use_hflip": False,
        "max_epochs": 10,
    }

    # Make the POST request to the train endpoint
    response = requests.post(f"{base_url}/train/", json=training_request)

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        print("\nTraining Started:")
        print(f"Status: {result['status']}")
        print(f"Detail: {result['detail']}")
        print(f"Experiment Path: {result['experiment_path']}")
        print(f"Model Weights Path: {result['model_weights_path']}")
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


def monitor_training_status(experiment_name, base_url="ws://localhost:8000"):
    """
    Monitor the training status using WebSocket connection.

    Args:
        experiment_name (str): Name of the experiment to monitor
        base_url (str): The WebSocket URL of your API server (default: ws://localhost:8000)
    """

    def on_message(ws, message):
        data = json.loads(message)
        print("\nTraining Progress:")
        print(f"Current Epoch: {data.get('current_epoch')}/{data.get('total_epochs')}")
        print(f"Current Loss: {data.get('current_loss'):.4f}")
        print(f"Best Loss: {data.get('best_loss'):.4f}")
        print(f"Status: {data.get('status')}")
        print(f"Detail: {data.get('detail')}")

        # Stop monitoring if training is completed or error occurred
        if data.get("status") in ["completed", "error"]:
            ws.close()

    def on_error(ws, error):
        print(f"Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        print("WebSocket connection closed")

    def on_open(ws):
        print(f"Connected to training monitor for experiment: {experiment_name}")

    # Create WebSocket connection
    ws = websocket.WebSocketApp(
        f"{base_url}/train/ws/{experiment_name}",
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open,
    )

    # Start WebSocket connection
    ws.run_forever()


def start_training_with_monitoring(experiment_name="test_experiment", base_url="http://localhost:8000"):
    """
    Start training and monitor its progress in parallel.

    Args:
        experiment_name (str): Name of the experiment
        base_url (str): The base URL of your API server
    """
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(
        target=monitor_training_status, args=(experiment_name, base_url.replace("http", "ws")), daemon=True
    )
    monitor_thread.start()

    # Give the WebSocket connection time to establish
    time.sleep(1)

    # Start the training
    result = test_train_endpoint(base_url)
    return result


# Example usage:
# result = start_training_with_monitoring()
