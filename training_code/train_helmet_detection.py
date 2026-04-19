"""
YOLOv8 Training Script for Helmet Detection
"""

from ultralytics import YOLO
import torch

def main():
    # Check GPU availability
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        device = 0
    else:
        print("GPU not available, using CPU (training will be slow)")
        device = 'cpu'

    # Load YOLOv8 Small model
    model = YOLO('yolov8s.pt')

    # Train the model
    # Reduced augmentation since dataset already augmented via Roboflow
    results = model.train(
        data='dataset/data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,  # Adjust based on GPU memory
        patience=20,  # Early stopping
        device=device,  # Auto-detect GPU or use CPU
        project='runs/detect',
        name='helmet_detection_v1',
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        seed=42,
        # Roboflow augmentation: Flip(Horizontal), Brightness(-15~+15%), 
        # Exposure(-8~+8%), Blur(0-1px)
        # Train augmentation: Minimal (avoid over-augmentation)
        # fliplr=0.0,  # Already flipped in Roboflow
        # flipud=0.0,  # No vertical flip
        # degrees=15,  # Slight rotation (not in Roboflow)
        # translate=0.1,  # Translation (not in Roboflow)
        # scale=0.5,  # Scale variation (not in Roboflow)
        # hsv_h=0.0,  # No HSV hue (brightness handled by Roboflow)
        # hsv_s=0.0,  # No saturation
        # hsv_v=0.0,  # No value/brightness (Roboflow -15~+15%)
        # mosaic=1.0,  # Keep mosaic for diversity
        # mixup=0.0,  # No mixup
        # copy_paste=0.0,  # No copy-paste
        # erasing=0.0  # No erasing
)

    # After training, validate on validation set
    print("\n" + "="*70)
    print("VALIDATION & EVALUATION ON VALIDATION SET")
    print("="*70)
    metrics = model.val(data='dataset/data.yaml')

    print("\n" + "="*70)
    print("EVALUATION METRICS")
    print("="*70)
    print(f"\n📊 Key Metrics:")
    print(f"   mAP50:        {metrics.box.map50:.4f}")
    print(f"   mAP50-95:     {metrics.box.map:.4f}")
    print(f"   Precision:    {metrics.box.p:.4f}")
    print(f"   Recall:       {metrics.box.r:.4f}")

    # Load best model for testing
    print("\n" + "="*70)
    print("TESTING BEST MODEL ON SAMPLE IMAGES")
    print("="*70)
    best_model = YOLO('runs/detect/runs/detect/helmet_detection_v1/weights/best.pt')

    # Test on validation images
    print("\n🔍 Running inference on validation set...")
    test_results = best_model.predict(
        source='dataset/valid/images',
        conf=0.45,
        save=True,
        save_txt=True,
        project='runs/detect/helmet_detection_v1',
        name='val_predictions'
    )

    print(f"\n✓ Tested on {len(test_results)} images")

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✅ Best model:     runs/detect/helmet_detection_v1/weights/best.pt")
    print(f"✅ Last model:     runs/detect/helmet_detection_v1/weights/last.pt")
    print(f"✅ Predictions:    runs/detect/helmet_detection_v1/val_predictions")
    print(f"\n📈 Training metrics saved in: runs/detect/helmet_detection_v1/")


if __name__ == '__main__':
    main()
