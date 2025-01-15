import supervision as sv
from src.transformers.base import Base

class ImageCreator:
    def __init__(self, model, source_video_path: str):
        self.model = model
        self.frame_generator = sv.get_video_frames_generator(
            source_video_path
        )

    def generate(self, image_generator: Base):
        frame = next(self.frame_generator)
        sv.plot_image(
            image_generator.infer(frame)
        )
