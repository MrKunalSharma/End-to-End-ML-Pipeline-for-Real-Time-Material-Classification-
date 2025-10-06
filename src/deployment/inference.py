import torch
import numpy as np
import cv2
from pathlib import Path
import json
import time
import onnxruntime as ort
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class MaterialClassifierInference:
    def __init__(self, model_path, model_type='onnx', device='cpu'):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to model file
            model_type: 'onnx' or 'torchscript'
            device: 'cpu' or 'cuda'
        """
        self.model_type = model_type
        self.device = device
        
        # Load metadata
        model_dir = Path(model_path).parent
        with open(model_dir / 'model_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.classes = self.metadata['classes']
        self.image_size = self.metadata['image_size']
        self.mean = np.array(self.metadata['normalization']['mean'], dtype=np.float32)
        self.std = np.array(self.metadata['normalization']['std'], dtype=np.float32)
        self.confidence_threshold = self.metadata['confidence_threshold']
        
        # Load model
        if model_type == 'onnx':
            self.session = ort.InferenceSession(str(model_path))
            self.input_name = self.session.get_inputs()[0].name
        else:  # torchscript
            self.model = torch.jit.load(str(model_path))
            self.model.eval()
            if device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.cuda()
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        # Resize
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 and normalize
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Transpose to CHW
        image = np.transpose(image, (2, 0, 1))
        
        # Add batch dimension and ensure float32
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        return image
    
    def predict(self, image):
        """
        Run inference on a single image
        
        Args:
            image: Input image (BGR numpy array)
            
        Returns:
            dict: Prediction results with class, confidence, and probabilities
        """
        # Preprocess
        input_tensor = self.preprocess(image)
        
        # Run inference
        start_time = time.time()
        
        if self.model_type == 'onnx':
            outputs = self.session.run(None, {self.input_name: input_tensor})[0]
            probabilities = self._softmax(outputs[0])
        else:  # torchscript
            with torch.no_grad():
                input_tensor = torch.from_numpy(input_tensor)
                if self.device == 'cuda' and torch.cuda.is_available():
                    input_tensor = input_tensor.cuda()
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()[0]
        
        inference_time = time.time() - start_time
        
        # Get prediction
        pred_idx = np.argmax(probabilities)
        confidence = probabilities[pred_idx]
        pred_class = self.classes[pred_idx]
        
        # Prepare result
        result = {
            'class': pred_class,
            'class_idx': int(pred_idx),
            'confidence': float(confidence),
            'probabilities': {cls: float(prob) for cls, prob in zip(self.classes, probabilities)},
            'inference_time': inference_time,
            'low_confidence': confidence < self.confidence_threshold
        }
        
        return result
    
    def _softmax(self, x):
        """Compute softmax values"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def predict_batch(self, images):
        """Run inference on multiple images"""
        results = []
        for image in images:
            results.append(self.predict(image))
        return results

def test_inference():
    """Test inference with a sample image"""
    # Create inference engine
    model_path = Path('models/deployed/model.onnx')
    if not model_path.exists():
        print("Model not found. Please export the model first.")
        return
    
    inference_engine = MaterialClassifierInference(model_path, model_type='onnx')
    
    # Test with a sample image
    test_image_path = None
    for class_dir in Path('data/processed/test').iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.jpg'))
            if images:
                test_image_path = images[0]
                break
    
    if test_image_path:
        image = cv2.imread(str(test_image_path))
        result = inference_engine.predict(image)
        
        print(f"Test image: {test_image_path}")
        print(f"Predicted class: {result['class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Inference time: {result['inference_time']*1000:.2f} ms")
        print(f"Low confidence warning: {result['low_confidence']}")

if __name__ == "__main__":
    test_inference()
