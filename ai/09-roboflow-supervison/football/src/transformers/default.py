import supervision as sv
from src.base import Base

class Model(Base):
    def __init__(self, model):
        super().__init__(model)

        self.box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
            thickness=2
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.from_hex(['#FF8C00', '#00BFFF', '#FF1493', '#FFD700']),
            text_color=sv.Color.from_hex('#000000')
        )

    def infer(self, frame, confidence=0.3):
        result = self.model.infer(frame, confidence=confidence)[0]
        detections = sv.Detections.from_inference(result)

        labels = [
            f"{class_name} {confidence:.2f}"
            for class_name, confidence
            in zip(detections['class_name'], detections.confidence)
        ]

        annotated_frame = frame.copy()
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        return annotated_frame
