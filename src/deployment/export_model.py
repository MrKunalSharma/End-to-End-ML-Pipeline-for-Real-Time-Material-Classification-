import torch
import torch.onnx
import sys
import os
from pathlib import Path
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.classifier import create_model
from src.utils.helpers import load_config, setup_logging

logger = setup_logging()
config = load_config()

def export_to_onnx(model_path, output_dir='models/deployed'):
    """Export PyTorch model to ONNX format"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size'])
    
    # Export to ONNX
    onnx_path = Path(output_dir) / 'model.onnx'
    logger.info(f"Exporting to ONNX: {onnx_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logger.info("ONNX export completed successfully")
    
    # Verify the model
    import onnxruntime as ort
    ort_session = ort.InferenceSession(str(onnx_path))
    outputs = ort_session.run(None, {'input': dummy_input.numpy()})
    logger.info(f"ONNX model verified. Output shape: {outputs[0].shape}")
    
    return str(onnx_path)

def export_to_torchscript(model_path, output_dir='models/deployed'):
    """Export PyTorch model to TorchScript format"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = create_model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, config['data']['image_size'], config['data']['image_size'])
    
    # Trace the model
    logger.info("Tracing model...")
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save TorchScript model
    torchscript_path = Path(output_dir) / 'model.pt'
    traced_model.save(str(torchscript_path))
    logger.info(f"TorchScript model saved to {torchscript_path}")
    
    # Verify the model
    loaded_model = torch.jit.load(str(torchscript_path))
    outputs = loaded_model(dummy_input)
    logger.info(f"TorchScript model verified. Output shape: {outputs.shape}")
    
    return str(torchscript_path)

def save_model_metadata(output_dir='models/deployed'):
    """Save model metadata for inference"""
    metadata = {
        'classes': config['data']['classes'],
        'image_size': config['data']['image_size'],
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        },
        'confidence_threshold': config['deployment']['confidence_threshold']
    }
    
    metadata_path = Path(output_dir) / 'model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logger.info(f"Model metadata saved to {metadata_path}")

def main():
    model_path = 'models/best_model.pth'
    
    if not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}. Train the model first!")
        return
    
    # Export to both formats
    if config['deployment']['onnx_export']:
        export_to_onnx(model_path)
    
    if config['deployment']['torchscript_export']:
        export_to_torchscript(model_path)
    
    # Save metadata
    save_model_metadata()
    
    logger.info("Model deployment preparation completed!")

if __name__ == "__main__":
    main()
