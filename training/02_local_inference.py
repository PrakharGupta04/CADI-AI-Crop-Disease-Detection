"""
CADI-AI Modified YOLO - Local Inference and Evaluation Script
=============================================================

This script:
1. Loads the trained model weights from Kaggle
2. Runs inference on single images or batches
3. Evaluates model on test dataset
4. Generates detailed evaluation metrics
5. Creates visualization plots
6. Saves results and comparisons

REQUIREMENTS:
- ultralytics
- opencv-python
- torch
- numpy
- pandas
- matplotlib
- pyyaml

USAGE:
  python local_inference.py --weights "modified_yolo_best.pt" --image "test_image.jpg"
  python local_inference.py --weights "modified_yolo_best.pt" --eval --dataset_config "data.yaml"
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: ENVIRONMENT SETUP
# ============================================================================

class CADIYOLOInference:
    """
    Inference and evaluation pipeline for CADI-AI Modified YOLO.
    
    Attributes:
        model: Loaded YOLO model
        device: torch device (cuda/cpu)
        class_names: List of class names
        metrics: Evaluation metrics dictionary
    """
    
    def __init__(self, weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize inference pipeline.
        
        Args:
            weights_path (str): Path to trained model weights
            device (str): Device to use (cuda/cpu)
        """
        self.device = torch.device(device)
        self.weights_path = weights_path
        self.model = None
        self.class_names = ['abiotic', 'insect', 'disease']
        self.metrics = {}
        
        print("="*80)
        print("CADI-AI MODIFIED YOLO - LOCAL INFERENCE")
        print("="*80)
        print(f"\nEnvironment Setup:")
        print(f"  Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        self.load_model()
    
    def load_model(self):
        """Load trained YOLO model from weights file."""
        print(f"\nLoading model from: {self.weights_path}")
        
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            self.model.to(self.device)
            print("✓ Model loaded successfully!")
            print(f"  Architecture: Modified YOLOv8")
            print(f"  Classes: {self.class_names}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    # ========================================================================
    # INFERENCE FUNCTIONS
    # ========================================================================
    
    def predict_single_image(self, image_path, conf_threshold=0.25, 
                            save_path=None, visualize=True):
        """
        Run inference on a single image.
        
        Args:
            image_path (str): Path to input image
            conf_threshold (float): Confidence threshold for detections
            save_path (str): Optional path to save annotated image
            visualize (bool): Whether to display results
        
        Returns:
            dict: Detection results with boxes, confidences, classes
        """
        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return None
        
        print(f"\nRunning inference on: {image_path}")
        
        try:
            # Run prediction
            results = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=0.45,
                device=self.device,
                verbose=False
            )
            
            result = results[0]
            
            # Parse results
            detections = {
                'image_path': image_path,
                'image_shape': result.orig_img.shape,
                'num_detections': len(result.boxes),
                'detections': []
            }
            
            # Extract box information
            if result.boxes is not None:
                for i, box in enumerate(result.boxes):
                    detection = {
                        'class_id': int(box.cls[0]),
                        'class_name': self.class_names[int(box.cls[0])],
                        'confidence': float(box.conf[0]),
                        'bbox': {
                            'x1': float(box.xyxy[0][0]),
                            'y1': float(box.xyxy[0][1]),
                            'x2': float(box.xyxy[0][2]),
                            'y2': float(box.xyxy[0][3]),
                        },
                        'width': float(box.xywh[0][2]),
                        'height': float(box.xywh[0][3]),
                    }
                    detections['detections'].append(detection)
            
            # Print results
            print(f"  ✓ Detections: {detections['num_detections']}")
            for det in detections['detections']:
                print(f"    - {det['class_name']}: {det['confidence']:.4f}")
            
            # Visualize if requested
            if visualize:
                self.visualize_results(result.orig_img, result)
            
            # Save annotated image if path provided
            if save_path:
                annotated_img = result.plot()
                cv2.imwrite(save_path, annotated_img)
                print(f"  ✓ Annotated image saved to: {save_path}")
            
            return detections
        
        except Exception as e:
            print(f"✗ Inference error: {e}")
            return None
    
    def predict_batch(self, image_dir, conf_threshold=0.25, 
                     save_dir=None, visualize=False):
        """
        Run inference on multiple images in a directory.
        
        Args:
            image_dir (str): Directory containing images
            conf_threshold (float): Confidence threshold
            save_dir (str): Directory to save annotated images
            visualize (bool): Whether to display results
        
        Returns:
            list: List of detection results for each image
        """
        if not os.path.isdir(image_dir):
            print(f"✗ Directory not found: {image_dir}")
            return None
        
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nRunning batch inference on: {image_dir}")
        
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"  Found {len(image_files)} images")
        
        results = []
        for i, image_file in enumerate(image_files, 1):
            image_path = os.path.join(image_dir, image_file)
            save_path = None
            
            if save_dir:
                save_path = os.path.join(save_dir, f'annotated_{image_file}')
            
            result = self.predict_single_image(
                image_path,
                conf_threshold=conf_threshold,
                save_path=save_path,
                visualize=visualize
            )
            
            if result:
                results.append(result)
            
            print(f"  [{i}/{len(image_files)}] Completed")
        
        return results
    
    # ========================================================================
    # EVALUATION FUNCTIONS
    # ========================================================================
    
    def evaluate_on_dataset(self, dataset_config_path, batch_size=16):
        """
        Evaluate model on full validation dataset.
        
        Args:
            dataset_config_path (str): Path to YOLO dataset config (data.yaml)
            batch_size (int): Batch size for evaluation
        
        Returns:
            dict: Evaluation metrics including mAP, precision, recall
        """
        if not os.path.exists(dataset_config_path):
            print(f"✗ Config file not found: {dataset_config_path}")
            return None
        
        print(f"\nEvaluating on dataset: {dataset_config_path}")
        
        try:
            # Run validation
            metrics = self.model.val(
                data=dataset_config_path,
                device=self.device,
                conf=0.25,
                iou=0.45,
                imgsz=640,
                batch=batch_size,
                verbose=True
            )
            
            # Store metrics
            self.metrics = {
                'model': 'Modified YOLO',
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'batch_size': batch_size,
            }
            
            # Extract key metrics
            if hasattr(metrics, 'results_dict'):
                self.metrics.update(metrics.results_dict)
            
            print("\n✓ Evaluation completed!")
            self._print_metrics()
            
            return metrics
        
        except Exception as e:
            print(f"✗ Evaluation error: {e}")
            return None
    
    def _print_metrics(self):
        """Print evaluation metrics in formatted table."""
        if not self.metrics:
            return
        
        print("\n" + "="*80)
        print("EVALUATION METRICS")
        print("="*80)
        
        # Key metrics to display
        key_metrics = [
            'metrics/mAP50(B)',
            'metrics/mAP50-95(B)',
            'metrics/precision(B)',
            'metrics/recall(B)',
        ]
        
        for metric_name in key_metrics:
            if metric_name in self.metrics:
                print(f"  {metric_name}: {self.metrics[metric_name]:.4f}")
    
    # ========================================================================
    # VISUALIZATION FUNCTIONS
    # ========================================================================
    
    def visualize_results(self, image, results):
        """
        Visualize detection results on image.
        
        Args:
            image: Original image (numpy array)
            results: YOLO results object
        """
        try:
            annotated = results.plot()
            
            # Display
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title('Detection Results')
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def create_comparison_plot(self, baseline_metrics, modified_metrics):
        """
        Create comparison plot between baseline and modified model.
        
        Args:
            baseline_metrics (dict): Baseline model metrics
            modified_metrics (dict): Modified model metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Baseline vs Modified YOLO Comparison', fontsize=16, fontweight='bold')
        
        # Metrics to compare
        metrics_to_compare = [
            ('mAP@0.5', 'metrics/mAP50(B)'),
            ('mAP@0.5:0.95', 'metrics/mAP50-95(B)'),
            ('Precision', 'metrics/precision(B)'),
            ('Recall', 'metrics/recall(B)'),
        ]
        
        for idx, (title, key) in enumerate(metrics_to_compare):
            ax = axes[idx // 2, idx % 2]
            
            baseline_val = baseline_metrics.get(key, 0)
            modified_val = modified_metrics.get(key, 0)
            
            models = ['Baseline\n(YOLOv8m)', 'Modified YOLO']
            values = [baseline_val, modified_val]
            colors = ['#FF6B6B', '#4ECDC4']
            
            bars = ax.bar(models, values, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}',
                       ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title(f'{title}', fontweight='bold')
            ax.set_ylim(0, max(baseline_val, modified_val) * 1.2)
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================
    
    def save_results(self, results, save_path):
        """
        Save inference results to JSON file.
        
        Args:
            results (dict/list): Results to save
            save_path (str): Path to save JSON file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        try:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"✓ Results saved to: {save_path}")
        except Exception as e:
            print(f"✗ Error saving results: {e}")
    
    def create_metrics_report(self, baseline_metrics, modified_metrics, save_path=None):
        """
        Create detailed metrics comparison report.
        
        Args:
            baseline_metrics (dict): Baseline model metrics
            modified_metrics (dict): Modified model metrics
            save_path (str): Optional path to save report
        
        Returns:
            str: Formatted report text
        """
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                    MODEL COMPARISON REPORT                                 ║
║                    CADI-AI Crop Disease Detection                          ║
╚════════════════════════════════════════════════════════════════════════════╝

EVALUATION DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
DEVICE: {self.device}

────────────────────────────────────────────────────────────────────────────
BASELINE MODEL: YOLOv8m (Pretrained, No Fine-tuning)
────────────────────────────────────────────────────────────────────────────

Mean Average Precision (mAP):
  • mAP@0.5:        {baseline_metrics.get('metrics/mAP50(B)', 0):.4f}
  • mAP@0.5:0.95:   {baseline_metrics.get('metrics/mAP50-95(B)', 0):.4f}

Precision & Recall:
  • Precision:      {baseline_metrics.get('metrics/precision(B)', 0):.4f}
  • Recall:         {baseline_metrics.get('metrics/recall(B)', 0):.4f}

────────────────────────────────────────────────────────────────────────────
MODIFIED MODEL: Enhanced YOLO with Attention & Multi-Scale Features
────────────────────────────────────────────────────────────────────────────

Mean Average Precision (mAP):
  • mAP@0.5:        {modified_metrics.get('metrics/mAP50(B)', 0):.4f}
  • mAP@0.5:0.95:   {modified_metrics.get('metrics/mAP50-95(B)', 0):.4f}

Precision & Recall:
  • Precision:      {modified_metrics.get('metrics/precision(B)', 0):.4f}
  • Recall:         {modified_metrics.get('metrics/recall(B)', 0):.4f}

────────────────────────────────────────────────────────────────────────────
IMPROVEMENT ANALYSIS
────────────────────────────────────────────────────────────────────────────

mAP@0.5 Improvement:
  • Baseline:       {baseline_metrics.get('metrics/mAP50(B)', 0):.4f}
  • Modified:       {modified_metrics.get('metrics/mAP50(B)', 0):.4f}
  • Improvement:    {(modified_metrics.get('metrics/mAP50(B)', 0) - baseline_metrics.get('metrics/mAP50(B)', 0)):.4f} 
                    ({((modified_metrics.get('metrics/mAP50(B)', 0) - baseline_metrics.get('metrics/mAP50(B)', 0)) / (baseline_metrics.get('metrics/mAP50(B)', 0) + 1e-6) * 100):.2f}%)

mAP@0.5:0.95 Improvement:
  • Baseline:       {baseline_metrics.get('metrics/mAP50-95(B)', 0):.4f}
  • Modified:       {modified_metrics.get('metrics/mAP50-95(B)', 0):.4f}
  • Improvement:    {(modified_metrics.get('metrics/mAP50-95(B)', 0) - baseline_metrics.get('metrics/mAP50-95(B)', 0)):.4f}
                    ({((modified_metrics.get('metrics/mAP50-95(B)', 0) - baseline_metrics.get('metrics/mAP50-95(B)', 0)) / (baseline_metrics.get('metrics/mAP50-95(B)', 0) + 1e-6) * 100):.2f}%)

Precision Improvement:
  • Baseline:       {baseline_metrics.get('metrics/precision(B)', 0):.4f}
  • Modified:       {modified_metrics.get('metrics/precision(B)', 0):.4f}
  • Improvement:    {(modified_metrics.get('metrics/precision(B)', 0) - baseline_metrics.get('metrics/precision(B)', 0)):.4f}
                    ({((modified_metrics.get('metrics/precision(B)', 0) - baseline_metrics.get('metrics/precision(B)', 0)) / (baseline_metrics.get('metrics/precision(B)', 0) + 1e-6) * 100):.2f}%)

Recall Improvement:
  • Baseline:       {baseline_metrics.get('metrics/recall(B)', 0):.4f}
  • Modified:       {modified_metrics.get('metrics/recall(B)', 0):.4f}
  • Improvement:    {(modified_metrics.get('metrics/recall(B)', 0) - baseline_metrics.get('metrics/recall(B)', 0)):.4f}
                    ({((modified_metrics.get('metrics/recall(B)', 0) - baseline_metrics.get('metrics/recall(B)', 0)) / (baseline_metrics.get('metrics/recall(B)', 0) + 1e-6) * 100):.2f}%)

════════════════════════════════════════════════════════════════════════════════
"""
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"✓ Report saved to: {save_path}")
        
        return report


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Main entry point for command-line usage."""
    
    parser = argparse.ArgumentParser(
        description='CADI-AI Modified YOLO Inference and Evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inference on single image
  python local_inference.py --weights "modified_yolo_best.pt" --image "test.jpg" --save "output.jpg"
  
  # Batch inference on directory
  python local_inference.py --weights "modified_yolo_best.pt" --image_dir "test_images/" --save_dir "results/"
  
  # Evaluation on full dataset
  python local_inference.py --weights "modified_yolo_best.pt" --eval --dataset_config "data.yaml"
  
  # Evaluation with comparison report
  python local_inference.py --weights "modified_yolo_best.pt" --eval --dataset_config "data.yaml" --baseline_config "baseline.yaml"
        """
    )
    
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to model weights file')
    parser.add_argument('--image', type=str,
                       help='Path to single image for inference')
    parser.add_argument('--image_dir', type=str,
                       help='Directory containing images for batch inference')
    parser.add_argument('--save', type=str,
                       help='Path to save annotated image')
    parser.add_argument('--save_dir', type=str,
                       help='Directory to save annotated images')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation on validation set')
    parser.add_argument('--dataset_config', type=str,
                       help='Path to YOLO dataset config (data.yaml)')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                       help='Confidence threshold for detections')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, 
                       default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--visualize', action='store_true',
                       help='Display detection visualizations')
    
    args = parser.parse_args()
    
    # Initialize inference pipeline
    inference = CADIYOLOInference(args.weights, device=args.device)
    
    # Single image inference
    if args.image:
        print("\n" + "="*80)
        print("SINGLE IMAGE INFERENCE")
        print("="*80)
        
        result = inference.predict_single_image(
            args.image,
            conf_threshold=args.conf_threshold,
            save_path=args.save,
            visualize=args.visualize
        )
        
        if result and args.save:
            inference.save_results(result, args.save.replace('.jpg', '.json').replace('.png', '.json'))
    
    # Batch inference
    elif args.image_dir:
        print("\n" + "="*80)
        print("BATCH IMAGE INFERENCE")
        print("="*80)
        
        results = inference.predict_batch(
            args.image_dir,
            conf_threshold=args.conf_threshold,
            save_dir=args.save_dir,
            visualize=args.visualize
        )
        
        if results and args.save_dir:
            inference.save_results(results, f'{args.save_dir}/batch_results.json')
    
    # Evaluation
    elif args.eval and args.dataset_config:
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        metrics = inference.evaluate_on_dataset(
            args.dataset_config,
            batch_size=args.batch_size
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("""
╔════════════════════════════════════════════════════════════════════════════╗
║        CADI-AI MODIFIED YOLO - LOCAL INFERENCE AND EVALUATION              ║
╚════════════════════════════════════════════════════════════════════════════╝

Usage:
  python local_inference.py --weights "path/to/weights.pt" [OPTIONS]

Options:
  --weights PATH              Path to model weights (REQUIRED)
  --image PATH               Single image for inference
  --image_dir PATH           Directory with images for batch inference
  --save PATH                Save annotated image
  --save_dir PATH            Directory to save annotated images
  --eval                     Run evaluation on dataset
  --dataset_config PATH      Path to data.yaml for evaluation
  --conf_threshold FLOAT     Confidence threshold (default: 0.25)
  --batch_size INT           Batch size (default: 16)
  --device DEVICE            Device to use: cuda or cpu
  --visualize               Display results

Examples:
  # Single image
  python local_inference.py --weights "modified_yolo_best.pt" --image "test.jpg"
  
  # Batch processing
  python local_inference.py --weights "modified_yolo_best.pt" --image_dir "images/" --save_dir "results/"
  
  # Full evaluation
  python local_inference.py --weights "modified_yolo_best.pt" --eval --dataset_config "data.yaml"

For detailed help:
  python local_inference.py --help
        """)
    else:
        main()
