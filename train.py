from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)
model.train(data='data.yaml',epochs=20, name='yolov8m')