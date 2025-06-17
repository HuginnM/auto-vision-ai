# AutoVisionAI Streamlit UI

A modern web interface for AutoVisionAI car segmentation pipeline, built with Streamlit.

## Features

### 🔍 Inference Page
- **File Upload**: Drag and drop or browse for image files (PNG, JPG, JPEG)
- **URL Input**: Provide image URLs for processing
- **Model Selection**: Choose between UNet, Fast-SCNN, and Mask R-CNN models
- **Real-time Results**: View inference results with detailed metrics
- **Result Export**: Download inference results as text files

### 🎯 Training Page
- **Hyperparameter Configuration**:
  - Batch size, epochs, early stopping patience
  - Data augmentation options (resize, random crop, horizontal flip)
- **Model Selection**: Choose architecture for training
- **Real-time Progress**: Live training progress with WebSocket updates
- **Loss Visualization**: Interactive loss graphs updated in real-time
- **Training Logs**: View detailed training output logs
- **Training Control**: Start/stop training with intuitive controls

## Quick Start

### 1. Install Dependencies
```bash
# Install Streamlit and other UI dependencies
pip install streamlit>=1.40.0 websocket-client>=1.8.0
```

### 2. Start the API Server
Make sure your AutoVisionAI API server is running:
```bash
# In one terminal, start the API server
python -m autovisionai.api.main
```

### 3. Launch the UI
```bash
# In another terminal, start the UI
python scripts/run_ui.py

# Or directly with streamlit
streamlit run src/autovisionai/ui/app.py
```

### 4. Access the Interface
Open your browser and navigate to: `http://localhost:8501`

## Configuration

### API Settings
- **API Base URL**: Configure the AutoVisionAI API endpoint (default: `http://localhost:8000`)
- The UI automatically detects if the API server is running

### Model Options
- **UNet**: Classic encoder-decoder architecture (~31M parameters)
- **Fast-SCNN**: Optimized for real-time inference (~1.1M parameters)
- **Mask R-CNN**: Instance segmentation with detection (~44M parameters)

## Architecture

```
src/autovisionai/ui/
├── app.py              # Main Streamlit application
├── pages/              # Page components
│   ├── inference_page.py    # Inference interface
│   └── training_page.py     # Training interface
├── utils.py            # Utility functions
└── README.md          # This file
```

## Key Components

### Main App (`app.py`)
- Application entry point
- Navigation between pages
- Global configuration and sidebar

### Inference Page (`pages/inference_page.py`)
- File upload and URL input handling
- API communication for inference requests
- Result visualization and export

### Training Page (`pages/training_page.py`)
- Training configuration form
- WebSocket connection for real-time updates
- Progress tracking and visualization
- Loss graph generation

### Utilities (`utils.py`)
- WebSocket client for training progress
- Formatting helpers
- Model information lookup

## WebSocket Integration

The training page uses WebSocket connections to receive real-time training updates:

```python
# WebSocket endpoint for training progress
ws://localhost:8000/train/ws/{experiment_name}
```

Progress updates include:
- Current epoch and total epochs
- Current loss and best loss
- Training status and logs
- Real-time metrics for visualization

## Error Handling

The UI includes comprehensive error handling for:
- **Network Issues**: Connection timeouts and API unavailability
- **File Handling**: Invalid file formats and upload errors
- **WebSocket**: Connection failures and message parsing errors
- **User Input**: Validation and sanitization

## Customization

### Adding New Models
Update the model selection in both pages:
```python
# In inference_page.py and training_page.py
model_name = st.selectbox(
    "Model",
    ["unet", "fast_scnn", "mask_rcnn", "your_new_model"],
    ...
)
```

### Styling
Streamlit themes can be customized in `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## Troubleshooting

### Common Issues

1. **API Connection Failed**
   - Ensure the API server is running on the correct port
   - Check the API Base URL in settings

2. **WebSocket Connection Issues**
   - Verify WebSocket endpoint is accessible
   - Check firewall settings for WebSocket connections

3. **File Upload Errors**
   - Ensure file size is within Streamlit limits (200MB default)
   - Check file format compatibility

4. **Training Progress Not Updating**
   - Verify WebSocket connection is established
   - Check browser console for JavaScript errors

### Performance Tips

- **Large Images**: Consider resizing images before upload for faster processing
- **Training Monitoring**: Use shorter update intervals for more responsive progress tracking
- **Memory Usage**: Close unused browser tabs when running long training sessions

## Development

### Running in Development Mode
```bash
# Enable debug mode and auto-reload
streamlit run src/autovisionai/ui/app.py --server.runOnSave true
```

### Testing
```bash
# Run UI tests (if implemented)
pytest tests/ui/
```

## Contributing

When adding new features to the UI:

1. Follow the existing page structure in `pages/`
2. Add utility functions to `utils.py`
3. Update this README with new features
4. Test with both real API and mock data
5. Ensure error handling is comprehensive
