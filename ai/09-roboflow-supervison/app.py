import cv2
import argparse
import supervision as sv
import shutil
import os
from supervision.assets import download_assets, VideoAssets
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
box_annotator = sv.BoxAnnotator()

def main(video_file_path):
    # Check if file exists
    if not os.path.exists(video_file_path):
        video_file_path = download_video_assets()

    print(f"Processing video file: {video_file_path}")
    frame_generator = sv.get_video_frames_generator(video_file_path)

    for i, frame in enumerate(frame_generator):
        result = model(frame, device="cuda")[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )

        imS = cv2.resize(annotated_frame, (960, 540))
        cv2.imshow("frame", imS)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

def download_video_assets():
    filename = download_assets(VideoAssets.VEHICLES_2)
    print(f"Downloaded video assets to {filename}")
    # Move to data folder
    shutil.move(filename, "data")
    return str(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video file")
    parser.add_argument("video_file_path", type=str, help="Path to video file", nargs='?', default='data/vehicles.mp4')
    args = parser.parse_args()
    main(args.video_file_path)
