from roboflow import Roboflow
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

API_KEY = os.environ.get("ROBOFLOW_API_KEY", "test")
WORKSPACE_NAME = os.environ.get("WORKSPACE_NAME", "roboflow-jvuqo")
PROJECT_NAME = os.environ.get("PROJECT_NAME", "football-players-detection-3zvbc")
PROJECT_VERSION = os.environ.get("PROJECT_VERSION", 12)

def main():
    HOME = os.environ.get("HOME_PATH", "/root")

    print(f"Ultralytics settings folder: {HOME}")
    if not os.path.exists(f"{HOME}/datasets"):
        os.makedirs(f"{HOME}/datasets")

    os.chdir(f"{HOME}/datasets")

    rf = Roboflow(api_key=API_KEY)

    project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
    version = project.version(12)
    dataset = version.download("yolov8")

    # !yolo task=detect mode=train model=yolov8x.pt data={dataset.location}/data.yaml batch=6 epochs=50 imgsz=1280 plots=True
    model = YOLO("yolov8x.pt")
    model.train(data=dataset.location + "/data.yaml", epochs=50, batch=6, imgsz=640, plots=True)

    display_images = [
        f'{HOME}/runs/detect/train/confusion_matrix.png',
        f'{HOME}/runs/detect/train/results.png',
        f'{HOME}/runs/detect/train/val_batch0_pred.jpg',
    ]

    for image in display_images:
        img = mpimg.imread(image)
        imgplot = plt.imshow(img)
        plt.show()

    # !yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml imgsz=1280
    myModel = YOLO(f"{HOME}/runs/detect/train/weights/best.pt")
    myModel.val(data=dataset.location + "/data.yaml", imgsz=640)

    my_workspace = rf.workspace(WORKSPACE_NAME)
    my_workspace.upload_dataset(
        dataset.location,
        PROJECT_NAME,
        num_workers=10,
    )

    my_project = my_workspace.project(PROJECT_NAME)
    new_version = my_project.generate_version(settings={
        "preprocessing": {
            "auto-orient": True,
            "resize": {"width": 640, "height": 640, "format": "Stretch to"},
        },
        "augmentation": {},
    })

    my_project.version(new_version).deploy(model_type="yolov8", model_path=f"{HOME}/runs/detect/train/")

if __name__ == "__main__":
    main()
