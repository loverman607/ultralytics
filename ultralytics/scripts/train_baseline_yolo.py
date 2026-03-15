from ultralytics import YOLO

# Baseline training (no LT modifications)
model = YOLO("yolo26n.pt")

model.train(
    data=r"c:\Users\ifeol\Documents\MACO\datasets\MACO\data.yaml",
    epochs=100,
    patience=20,
    workers=0,
)
