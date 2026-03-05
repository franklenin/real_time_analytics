"""Video processing pipeline for real-time soccer analytics."""

import cv2
import numpy as np
from ultralytics import YOLO

from .team_assigner import TeamAssigner

# BGR colors used for each team's annotations
_TEAM_COLORS: dict[int, tuple[int, int, int]] = {
    1: (0, 0, 220),   # Red
    2: (220, 0, 0),   # Blue
}

# Minimum detection confidence threshold
_CONF_THRESHOLD = 0.5

# YOLO class index for "person"
_PERSON_CLASS = 0


class VideoProcessor:
    """Processes a soccer video with YOLO-v8 detection and team counting.

    Args:
        model_path: Path to the YOLO-v8 model weights file.  Ultralytics
            will download the weights automatically on first use when a
            standard model name such as ``yolov8n.pt`` is given.
    """

    def __init__(self, model_path: str = "yolov8n.pt") -> None:
        self.model = YOLO(model_path)
        self.team_assigner = TeamAssigner()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_players(self, frame: np.ndarray) -> dict[int, list[float]]:
        """Run YOLO inference and return confident person detections.

        Returns:
            Mapping of detection index → [x1, y1, x2, y2].
        """
        results = self.model(frame, classes=[_PERSON_CLASS], verbose=False)
        detections: dict[int, list[float]] = {}

        if results and results[0].boxes is not None:
            for i, box in enumerate(results[0].boxes):
                conf = float(box.conf[0])
                if conf >= _CONF_THRESHOLD:
                    detections[i] = box.xyxy[0].tolist()

        return detections

    def _annotate_frame(
        self,
        frame: np.ndarray,
        player_detections: dict[int, list[float]],
        team_counts: dict[int, int],
    ) -> np.ndarray:
        """Draw bounding boxes, team labels, and count overlay on a frame."""
        annotated = frame.copy()

        for player_id, bbox in player_detections.items():
            x1, y1, x2, y2 = (int(v) for v in bbox)
            team_id = self.team_assigner.get_team(frame, bbox, player_id)
            color = _TEAM_COLORS.get(team_id, (0, 200, 0))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                f"T{team_id}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        # Semi-transparent scoreboard in the top-left corner
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (230, 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

        for idx, (team_id, color) in enumerate(_TEAM_COLORS.items()):
            count = team_counts.get(team_id, 0)
            cv2.putText(
                annotated,
                f"Team {team_id}: {count} players",
                (20, 38 + idx * 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, video_path: str, output_path: str | None = None) -> None:
        """Analyse a soccer video frame by frame.

        Detects players with YOLO-v8, assigns them to one of two teams by
        jersey color, and displays a live count for each team.  Press **q**
        to quit early.

        Args:
            video_path: Path to the input video file.
            output_path: Optional path where the annotated video is saved.
                If *None*, the video is shown in a window but not saved.

        Raises:
            ValueError: If the video file cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer: cv2.VideoWriter | None = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        team_initialized = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                player_detections = self._detect_players(frame)

                # Fit the team-color model on the first frame with ≥2 players
                if not team_initialized and len(player_detections) >= 2:
                    self.team_assigner.fit(frame, player_detections)
                    team_initialized = True

                # Count players per team
                team_counts: dict[int, int] = {1: 0, 2: 0}
                for player_id, bbox in player_detections.items():
                    team_id = self.team_assigner.get_team(frame, bbox, player_id)
                    team_counts[team_id] += 1

                annotated_frame = self._annotate_frame(
                    frame, player_detections, team_counts
                )

                if writer is not None:
                    writer.write(annotated_frame)

                cv2.imshow("Soccer Analytics", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                frame_num += 1

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
