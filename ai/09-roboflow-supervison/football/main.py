import argparse
import gdown
import os
import supervision as sv
from inference import get_model
from src.image_creator import ImageCreator
from src.video_creator import VideoCreator
from src.transformers.default import Model as DefaultModel
from src.transformers.styled import Model as StyledModel
from src.transformers.player_tracking import Model as PlayerTrackingModel
from src.transformers.base import Base
from src.train.split_into_teams_html import main as split_into_teams_html

# os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "test")
PLAYER_DETECTION_MODEL_ID = os.environ.get("PLAYER_DETECTION_MODEL_ID", "football-players-detection-3zvbc/11")

DOWNLOAD_DIR = "./data"
DOWNLOAD_VIDEO_FILES = [
    { "id": "12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF", "name": "0bfacc_0.mp4" },
    { "id": "19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf", "name": "2e57b9_0.mp4" },
    { "id": "1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-", "name": "08fd33_0.mp4" },
    { "id": "1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU", "name": "573e61_0.mp4" },
    { "id": "1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu", "name": "121364_0.mp4" }
]

def download_videos():
    for file in DOWNLOAD_VIDEO_FILES:
        file_path = f"{DOWNLOAD_DIR}/{file['name']}"

        if not os.path.exists(file_path):
            gdown.download(f"https://drive.google.com/uc?id={file['id']}", file_path, quiet=False)

def main():
    download_videos()

    PLAYER_DETECTION_MODEL = get_model(model_id=PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    SOURCE_VIDEO_PATH = f"{DOWNLOAD_DIR}/{DOWNLOAD_VIDEO_FILES[-1]['name']}"
    DESTINATION_VIDEO_PATH = f"{DOWNLOAD_DIR}/annotated_{DOWNLOAD_VIDEO_FILES[-1]['name']}"

    parser = argparse.ArgumentParser(description="Process video file")

    parser.add_argument("--image", action=argparse.BooleanOptionalAction, help="Show image", default=True)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction, help="Show video", default=False)

    parser.add_argument("--default", action=argparse.BooleanOptionalAction, help="Use default model", default=True)
    parser.add_argument("--styled", action=argparse.BooleanOptionalAction, help="Use styled model", default=False)
    parser.add_argument("--player-tracking", action=argparse.BooleanOptionalAction, help="Use player tracking model", default=False)
    parser.add_argument("--split-into-teams-html", action=argparse.BooleanOptionalAction, help="Generate split into teams html", default=False)

    args = parser.parse_args()

    if args.train_split_into_teams_hmtl:
        split_into_teams_html(PLAYER_DETECTION_MODEL, SOURCE_VIDEO_PATH)
        return

    transform_frame_generator = None

    if args.styled:
        transform_frame_generator = StyledModel(PLAYER_DETECTION_MODEL)
    elif args.player_tracking:
        transform_frame_generator = PlayerTrackingModel(PLAYER_DETECTION_MODEL)
    elif args.default:
        transform_frame_generator = DefaultModel(PLAYER_DETECTION_MODEL)

    if transform_frame_generator is None:
        parser.print_help()
        return

    if args.video:
        main_video(args, PLAYER_DETECTION_MODEL, SOURCE_VIDEO_PATH, DESTINATION_VIDEO_PATH, transform_frame_generator)
    elif args.image:
        main_image(args, PLAYER_DETECTION_MODEL, SOURCE_VIDEO_PATH, transform_frame_generator)
    else:
        parser.print_help()

def main_image(args, model, source_video_path: str, transform_frame_generator: Base):
    image_creator = ImageCreator(model, source_video_path)
    image_creator.generate(transform_frame_generator)

def main_video(args, model, source_video_path: str, destination_video_path: str, transform_frame_generator: Base):
    video_creator = VideoCreator(model, source_video_path, destination_video_path)
    video_creator.generate(transform_frame_generator)

if __name__ == "__main__":
    main()
