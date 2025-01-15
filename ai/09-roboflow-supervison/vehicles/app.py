import cv2
import argparse
import supervision as sv
import shutil
import os
import numpy as np
from supervision.assets import download_assets, VideoAssets
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

tracker = sv.ByteTrack(minimum_consecutive_frames=3)
tracker.reset()

smother = sv.DetectionsSmoother()

POLYGON = np.array([[7, 1611], [1421, 591], [1534, 433], [1557, 363], [1804, 363], [1854, 479], [2001, 607], [3837, 1441], [3829, 2145], [19, 2149]])
VEHICLE_CLASSES = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
]

LINE_1_START_POSITION = sv.Point(88, 1720)
LINE_1_END_POSITION = sv.Point(1847, 1723)
LINE_2_START_POSITION = sv.Point(2422, 1739)
LINE_2_END_POSITION = sv.Point(3810, 1565)

LINE_ZONE_1 = sv.LineZone(start=LINE_1_START_POSITION, end=LINE_1_END_POSITION, triggering_anchors=(sv.Position.BOTTOM_CENTER,))
LINE_ZONE_2 = sv.LineZone(start=LINE_2_START_POSITION, end=LINE_2_END_POSITION, triggering_anchors=(sv.Position.BOTTOM_CENTER,))

# Filter list of classes to only include vehicles
FILTERED_CLASSES = [name for name in model.names if model.names[name] in VEHICLE_CLASSES]

polygon_zone = sv.PolygonZone(polygon=POLYGON, triggering_anchors=(sv.Position.CENTER,))
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_scale=1.3, text_thickness=2)
trace_annotator = sv.TraceAnnotator(trace_length=60)
line_zone_annotator = sv.LineZoneAnnotator(
    text_scale=1.2,
    text_orient_to_line=True
)
line_zone_annotator_multiclass = sv.LineZoneAnnotatorMulticlass(
    text_scale=1.2,
    text_thickness=2,
    table_margin=20
)

def main(video_file_path):
    print(f"Processing video file: {video_file_path}")
    frame_generator = sv.get_video_frames_generator(video_file_path)

    for i, frame in enumerate(frame_generator):
        result = model(frame, device="cuda", verbose=False, imgsz=1280)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections[np.isin(detections.class_id, FILTERED_CLASSES)]
        detections = tracker.update_with_detections(detections)
        detections = smother.update_with_detections(detections)

        labels = [
            f"#{tracker_id} - {model.names[class_id]} ({confidence:.2f})"
            for tracker_id, class_id, confidence
                in zip(detections.tracker_id, detections.class_id, detections.confidence)
        ]

        LINE_ZONE_1.trigger(detections=detections)
        LINE_ZONE_2.trigger(detections=detections)

        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(
            scene=annotated_frame,
            polygon=POLYGON,
            color=sv.Color.RED,
            thickness=2,
        )
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
        )
        annotated_frame = line_zone_annotator.annotate(
            annotated_frame,
            line_counter=LINE_ZONE_1
        )
        annotated_frame = line_zone_annotator.annotate(
            annotated_frame,
            line_counter=LINE_ZONE_2
        )
        annotated_frame = line_zone_annotator_multiclass.annotate(
            annotated_frame,
            line_zones=[LINE_ZONE_1, LINE_ZONE_2],
            line_zone_labels=["Left", "Right"],
        )

        imS = cv2.resize(annotated_frame, (960, 540))
        cv2.imshow("frame", imS)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

def check_video_file_path(video_file_path):
    if video_file_path is None:
        return download_video_assets()
    if not os.path.exists(video_file_path):
        raise FileNotFoundError(f"File not found: {video_file_path}")
    return video_file_path

def download_video_assets():
    assetToDownload = VideoAssets.VEHICLES

    if not os.path.exists("data"):
        os.makedirs("data")

    if not os.path.exists("data/" + assetToDownload.value):
        filename = download_assets(assetToDownload)
        shutil.move(filename, "data")

    return str("data/" + assetToDownload.value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video file")
    parser.add_argument("video_file_path", type=str, help="Path to video file. Use asset by default.", nargs='?')
    args = parser.parse_args()
    main(check_video_file_path(args.video_file_path))
