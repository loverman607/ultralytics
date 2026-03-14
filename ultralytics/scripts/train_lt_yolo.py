from ultralytics import YOLO
from detect.train import LTTrainer  # your custom trainer

# instantiate YOLO with pretrained weights
model = YOLO("weights/yolo26n.pt")

# train using your LTTrainer which uses LTDetectionModel + TailBalancedLoss internally
model.train(
    data="data/coco8.yaml",
    epochs=20,
    trainer=LTTrainer,
    freeze=10
)