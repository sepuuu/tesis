from ultralytics import YOLO

# Load a model
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs

results = model.train(data='config.yaml', epochs=3)