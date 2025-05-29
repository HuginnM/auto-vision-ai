import asyncio
import json
import logging

import requests
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

logger = logging.getLogger(__name__)


def test_train_endpoint(base_url="http://127.0.0.1:8000"):
    """
    Test the train endpoint with a sample training request.

    Args:
        base_url (str): The base URL of your API server (default: http://localhost:8000)

    Returns:
        dict: The response from the training endpoint
    """
    training_request = {
        "experiment_name": "test_experiment",
        "model_name": "fast_scnn",
        "batch_size": 4,
        "epoch_patience": 2,
        "use_resize": False,
        "use_random_crop": True,
        "use_hflip": True,
        "max_epochs": 2,
    }

    # Make the POST request to the train endpoint
    response = requests.post(f"{base_url}/train/", json=training_request)

    if response.status_code == 200:
        result = response.json()
        print("\nTraining Result:")
        print(f"Status: {result['status']}")
        print(f"Detail: {result['detail']}")
        print(f"Experiment Path: {result['experiment_path']}")
        print(f"Model Weights Path: {result['model_weights_path']}")
        logger.info(f"Training started: {result}")
        return result
    else:
        logger.error(f"Error: {response.status_code}")
        logger.error(response.text)
        return None


async def monitor_training(experiment_name: str):
    """Monitor training progress via WebSocket."""
    uri = f"ws://127.0.0.1:8000/train/ws/{experiment_name}"
    logger.info(f"Connecting to {uri}")

    try:
        async with websockets.connect(uri) as websocket:
            logger.info(f"Connected to training monitor for experiment: {experiment_name}")

            while True:
                try:
                    data = await websocket.recv()
                    try:
                        progress = json.loads(data)
                        logger.info("Training progress: %s", progress)

                        if progress.get("status") in ["completed", "error"]:
                            logger.info(f"Terminal status '{progress.get('status')}' received. Exiting monitor.")
                            break
                    except json.JSONDecodeError:
                        logger.warning("Received non-JSON: %s", data)
                except websockets.ConnectionClosed:
                    logger.warning("WebSocket closed by server.")
                    break

    except Exception as e:
        logger.error(f"Error connecting to WebSocket: {e}", exc_info=True)


async def start_training_with_monitoring():
    """Start training and monitor progress concurrently."""
    monitoring_task = asyncio.create_task(monitor_training("test_experiment"))

    # Give WebSocket time to connect
    await asyncio.sleep(1)
    training_result = await asyncio.to_thread(test_train_endpoint)
    await monitoring_task
    return training_result


if __name__ == "__main__":
    asyncio.run(start_training_with_monitoring())
