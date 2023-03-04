from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
model.train(data='data_test.yaml',epochs=20, name='yolov8n_test')