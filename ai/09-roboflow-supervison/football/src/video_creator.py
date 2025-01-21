import supervision as sv
from tqdm import tqdm
from src.base import Base

class VideoCreator:
    def __init__(self, model, source_video_path: str, destination_video_path: str):
        self.model = model
        self.source_video_path = source_video_path
        self.destination_video_path = destination_video_path

        self.frame_generator = sv.get_video_frames_generator(
            source_video_path
        )

    def generate(self, image_generator: Base):
        video_info = sv.VideoInfo.from_video_path(self.source_video_path)
        video_sink = sv.VideoSink(self.destination_video_path, video_info=video_info)

        with video_sink:
            for frame in tqdm(self.frame_generator, total=video_info.total_frames, desc='processing video'):
                video_sink.write_frame(
                    image_generator.infer(frame)
                )

        print(f"Video saved to {self.destination_video_path}")
