from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model.train(data='data.yaml',epochs=30, name='yolov8n', batch=16, save_period=5)

model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
model.train(data='data.yaml',epochs=30, name='yolov8s', batch=16, save_period=5)

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model.train(data='data.yaml',epochs=50, name='yolov8n', batch=8, save_period=5)

model = YOLO("yolov8s.pt")  # load a pretrained model (recommended for training)
model.train(data='data.yaml',epochs=50, name='yolov8s', batch=8, save_period=5)