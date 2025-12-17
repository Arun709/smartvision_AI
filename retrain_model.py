from ultralytics import YOLO

def main():
    print("🚀 Starting new training session...")
    
    # 1. Load the pre-trained YOLOv8 Nano model (it learns faster)
    model = YOLO('yolov8n.pt')
    
    # 2. Start training using your new dataset
    # Make sure 'data.yaml' is the path to the Roboflow YAML file
    results = model.train(
        data='Distracted-Driver-Detection-1/data.yaml',
        epochs=50,  # 50 is a good starting point for a decent model
        imgsz=640,
        name='distracted_driver_yolov8'
    )
    
    print("✅ Training complete! New model saved in 'runs/detect/distracted_driver_yolov8'")
    print("➡️ Copy 'runs/detect/distracted_driver_yolov8/weights/best.pt' to your 'models/' folder to use it.")

if __name__ == '__main__':
    main()
