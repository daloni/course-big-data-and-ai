import supervision as sv
from src.base import Base

class Model(Base):
    def __init__(self, model):
        super().__init__(model)

        self.BALL_ID = 0

        self.ellipse_annotator = sv.EllipseAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000'),
            text_position=sv.Position.BOTTOM_CENTER
        )
        self.triangle_annotator = sv.TriangleAnnotator(
            color=sv.Color.from_hex('#FFD700'),
            base=25,
            height=21,
            outline_thickness=1
        )

        self.tracker = sv.ByteTrack()
        self.tracker.reset()

    def infer(self, frame, confidence=0.3):
        result = self.model.infer(frame, confidence=0.3)[0]
        detections = sv.Detections.from_inference(result)

        ball_detections = detections[detections.class_id == self.BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        all_detections = detections[detections.class_id != self.BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
        all_detections.class_id -= 1
        all_detections = self.tracker.update_with_detections(detections=all_detections)

        labels = [
            f"#{tracker_id}"
            for tracker_id
            in all_detections.tracker_id
        ]

        annotated_frame = frame.copy()
        annotated_frame = self.ellipse_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections)
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=all_detections,
            labels=labels)
        annotated_frame = self.triangle_annotator.annotate(
            scene=annotated_frame,
            detections=ball_detections)

        return annotated_frame
