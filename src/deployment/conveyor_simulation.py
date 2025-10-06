import cv2
import time
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.deployment.inference import MaterialClassifierInference
from src.utils.helpers import setup_logging, load_config

logger = setup_logging()
config = load_config()

class ConveyorSimulation:
    def __init__(self, inference_engine, frame_source='folder'):
        """
        Initialize conveyor simulation
        
        Args:
            inference_engine: MaterialClassifierInference instance
            frame_source: 'folder' or 'video'
        """
        self.inference_engine = inference_engine
        self.frame_source = frame_source
        self.frame_interval = config['simulation']['frame_interval']
        self.results = []
        self.misclassified_queue = []
        
        # Setup CSV logging
        self.csv_file = Path('results') / f'simulation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        self.csv_headers = ['timestamp', 'frame_id', 'image_path', 'predicted_class', 
                           'confidence', 'inference_time_ms', 'low_confidence_flag']
        
        # Initialize CSV file
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.csv_headers)
    
    def load_frames(self, source_path):
        """Load frames from folder or video"""
        frames = []
        paths = []
        
        if self.frame_source == 'folder':
            # Load all images from test folder
            test_dir = Path(source_path)
            for class_dir in test_dir.iterdir():
                if class_dir.is_dir():
                    for img_path in class_dir.glob('*.jpg'):
                        image = cv2.imread(str(img_path))
                        if image is not None:
                            frames.append(image)
                            paths.append(str(img_path))
        else:
            # Load frames from video
            cap = cv2.VideoCapture(str(source_path))
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                paths.append(f"frame_{frame_count}")
                frame_count += 1
            cap.release()
        
        return frames, paths
    
    def simulate_conveyor(self, source_path='data/processed/test'):
        """Run the conveyor simulation"""
        logger.info("Starting conveyor simulation...")
        
        # Load frames
        frames, paths = self.load_frames(source_path)
        if not frames:
            logger.error("No frames loaded!")
            return
        
        logger.info(f"Loaded {len(frames)} frames for processing")
        
        # Process each frame
        for frame_id, (frame, path) in enumerate(zip(frames, paths)):
            start_time = time.time()
            
            # Run inference
            result = self.inference_engine.predict(frame)
            
            # Log to console
            logger.info(f"\n{'='*50}")
            logger.info(f"Frame {frame_id + 1}/{len(frames)}")
            logger.info(f"Image: {Path(path).name}")
            logger.info(f"Predicted: {result['class']} (confidence: {result['confidence']:.4f})")
            logger.info(f"Inference time: {result['inference_time']*1000:.2f} ms")
            
            if result['low_confidence']:
                logger.warning(f"⚠️  LOW CONFIDENCE DETECTION! Confidence: {result['confidence']:.4f}")
            
            # Save to CSV
            self._log_to_csv(frame_id, path, result)
            
            # Store result
            self.results.append({
                'frame_id': frame_id,
                'path': path,
                'result': result,
                'timestamp': datetime.now().isoformat()
            })
            
            # Check for manual override (bonus feature)
            if config['simulation']['enable_manual_override'] and result['low_confidence']:
                self._handle_manual_override(frame, path, result)
            
            # Simulate frame interval
            elapsed = time.time() - start_time
            sleep_time = max(0, self.frame_interval - elapsed)
            time.sleep(sleep_time)
        
        # Summary
        self._print_summary()
        self._save_final_report()
    
    def _log_to_csv(self, frame_id, path, result):
        """Log result to CSV file"""
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                frame_id,
                path,
                result['class'],
                f"{result['confidence']:.4f}",
                f"{result['inference_time']*1000:.2f}",
                result['low_confidence']
            ])
    
    def _handle_manual_override(self, frame, path, result):
        """Handle manual override for misclassifications"""
        self.misclassified_queue.append({
            'path': path,
            'predicted': result['class'],
            'confidence': result['confidence'],
            'timestamp': datetime.now().isoformat()
        })
        logger.info(f"Added to retraining queue: {Path(path).name}")
    
    def _print_summary(self):
        """Print simulation summary"""
        logger.info(f"\n{'='*60}")
        logger.info("SIMULATION SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total frames processed: {len(self.results)}")
        
        # Calculate statistics
        confidences = [r['result']['confidence'] for r in self.results]
        inference_times = [r['result']['inference_time'] for r in self.results]
        low_conf_count = sum(1 for r in self.results if r['result']['low_confidence'])
        
        logger.info(f"Average confidence: {np.mean(confidences):.4f}")
        logger.info(f"Average inference time: {np.mean(inference_times)*1000:.2f} ms")
        logger.info(f"Low confidence detections: {low_conf_count} ({low_conf_count/len(self.results)*100:.1f}%)")
        
        # Class distribution
        class_counts = {}
        for r in self.results:
            cls = r['result']['class']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        logger.info("\nClass distribution:")
        for cls, count in sorted(class_counts.items()):
            logger.info(f"  {cls}: {count} ({count/len(self.results)*100:.1f}%)")
        
        logger.info(f"\nResults saved to: {self.csv_file}")
    
    def _save_final_report(self):
        """Save comprehensive final report"""
        report = {
            'simulation_info': {
                'timestamp': datetime.now().isoformat(),
                'total_frames': len(self.results),
                'frame_interval': self.frame_interval,
                'model_type': self.inference_engine.model_type
            },
            'statistics': {
                'average_confidence': float(np.mean([r['result']['confidence'] for r in self.results])),
                'average_inference_time_ms': float(np.mean([r['result']['inference_time'] for r in self.results]) * 1000),
                'low_confidence_count': sum(1 for r in self.results if r['result']['low_confidence'])
            },
            'misclassified_queue': self.misclassified_queue
        }
        
        report_path = Path('results') / f'simulation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        logger.info(f"Final report saved to: {report_path}")

def main():
    """Run conveyor simulation"""
    # Check if model exists
    model_path = Path('models/deployed/model.onnx')
    if not model_path.exists():
        logger.error("Deployed model not found. Please export the model first.")
        return
    
    # Create inference engine
    inference_engine = MaterialClassifierInference(
        model_path, 
        model_type='onnx',
        device='cpu'
    )
    
    # Create and run simulation
    simulation = ConveyorSimulation(inference_engine)
    simulation.simulate_conveyor()

if __name__ == "__main__":
    main()
