"""Entry point for real-time soccer video analytics."""

import argparse

from analytics.video_processor import VideoProcessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse a soccer video using YOLO-v8 and count players per team."
    )
    parser.add_argument("video", help="Path to the input soccer video file.")
    parser.add_argument(
        "--output",
        "-o",
        metavar="OUTPUT",
        default=None,
        help="Path where the annotated output video is saved (optional).",
    )
    parser.add_argument(
        "--model",
        "-m",
        metavar="MODEL",
        default="yolov8n.pt",
        help="YOLO model weights file (default: yolov8n.pt).",
    )

    args = parser.parse_args()

    processor = VideoProcessor(model_path=args.model)
    processor.process(args.video, output_path=args.output)


if __name__ == "__main__":
    main()
