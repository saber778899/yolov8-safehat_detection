from ultralytics import YOLO

model = YOLO('best.pt')

model.predict('test14.jpg', save=True, classes=[0, 2])

