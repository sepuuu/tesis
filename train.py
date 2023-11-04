from ultralytics import YOLO
import ultralytics
# Load a model
model = YOLO('modelos/yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
#results = model.train(data='config.yaml', epochs=3 )

results = model.train(data='config.yaml', epochs=10, imgsz=640, device="cpu")



