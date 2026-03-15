from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import LTTrainer  # your custom trainer
import multiprocessing


def main():
    # instantiate YOLO with pretrained weights
    model = YOLO("yolo26n.pt")

    # train using your LTTrainer which uses LTDetectionModel + TailBalancedLoss internally
    model.train(
        data=r"c:\Users\ifeol\Documents\MACO\datasets\MACO\data.yaml",
        epochs=500,
        trainer=LTTrainer,
        warmup_epochs=3,
        batch=16,
        workers=4,
        patience=50,
    )

    # Alternative resume example (commented out)
    # model = YOLO(r"C:\Users\ifeol\Documents\MACO\yoloresearch\runs\detect\train\weights\last.pt")
    # model.train(
    #     data=r"c:\Users\ifeol\Documents\MACO\datasets\MACO\data.yaml",
    #     epochs=100,
    #     trainer=LTTrainer,
    #     workers=0,
    #     patience=20,
    #     resume=True,
    # )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()

