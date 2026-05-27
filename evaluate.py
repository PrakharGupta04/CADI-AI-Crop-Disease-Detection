"""
Evaluate model on validation dataset with annotations
Requires data/
  ├── images/
  └── labels/

Run: python evaluate.py --dataset_path data/
"""

import argparse
import os
from ultralytics import YOLO
import yaml

def main():
    parser = argparse.ArgumentParser(description='Model Evaluation')
    parser.add_argument('--dataset_path', type=str, default='data/',
                       help='Path to dataset folder')
    parser.add_argument('--weights', type=str,
                       default='weights/modified_yolo_best.pt')
    
    args = parser.parse_args()
    
    # Create data.yaml if it doesn't exist
    dataset_path = args.dataset_path
    
    if not os.path.exists(f"{dataset_path}/data.yaml"):
        # Create default config
        config = {
            'path': os.path.abspath(dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/val',
            'nc': 3,
            'names': ['abiotic', 'insect', 'disease']
        }
        
        with open(f"{dataset_path}/data.yaml", 'w') as f:
            yaml.dump(config, f)
        
        print(f"Created data.yaml")
    
    # Load model
    print(f"Loading model...")
    model = YOLO(args.weights)
    
    # Evaluate
    print(f"Evaluating on dataset...")
    metrics = model.val(
        data=f"{dataset_path}/data.yaml",
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.45
    )
    
    print(f"\nEvaluation Results:")
    print(f"  mAP@0.5: {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:0.95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
    print(f"  Precision: {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall: {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")

if __name__ == '__main__':
    main()