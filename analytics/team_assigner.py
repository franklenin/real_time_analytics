"""Team assignment using jersey color clustering."""

import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    """Assigns detected players to teams based on jersey color.

    Uses K-means clustering on per-player jersey colors to group all players
    into exactly two teams on the first frame that has enough detections.
    Subsequent frames reuse the fitted cluster model for consistency.
    """

    def __init__(self) -> None:
        self.team_colors: dict[int, np.ndarray] = {}
        self.player_team_dict: dict[int, int] = {}
        self.kmeans: KMeans | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_jersey_color(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Return the dominant jersey color for a player crop.

        The top half of the bounding box is used as a proxy for the jersey
        area.  Within that crop a two-cluster K-means is fitted and the
        cluster that does *not* dominate the four corners (assumed to be
        background) is returned as the jersey color.
        """
        x1, y1, x2, y2 = (int(v) for v in bbox)
        mid_y = (y1 + y2) // 2
        jersey_crop = frame[y1:mid_y, x1:x2]

        if jersey_crop.size == 0:
            return np.zeros(3, dtype=np.float64)

        pixels = jersey_crop.reshape(-1, 3).astype(np.float64)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=3, random_state=0)
        kmeans.fit(pixels)

        # Identify background cluster from the four corner pixels
        h, w = jersey_crop.shape[:2]
        corners = np.array(
            [
                jersey_crop[0, 0],
                jersey_crop[0, w - 1],
                jersey_crop[h - 1, 0],
                jersey_crop[h - 1, w - 1],
            ],
            dtype=np.float64,
        )
        corner_labels = kmeans.predict(corners)
        bg_cluster = int(np.bincount(corner_labels).argmax())
        jersey_cluster = 1 - bg_cluster

        return kmeans.cluster_centers_[jersey_cluster]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, frame: np.ndarray, player_detections: dict[int, list[float]]) -> None:
        """Fit the two-team K-means model on the current frame's detections.

        This should be called once on the first frame that contains at least
        two player detections so that the cluster centers can be established.

        Args:
            frame: The video frame as a BGR NumPy array.
            player_detections: Mapping of player ID → [x1, y1, x2, y2] bbox.
        """
        if len(player_detections) < 2:
            return

        jersey_colors = [
            self._get_jersey_color(frame, bbox)
            for bbox in player_detections.values()
        ]

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        kmeans.fit(jersey_colors)

        self.kmeans = kmeans
        self.team_colors = {
            1: kmeans.cluster_centers_[0],
            2: kmeans.cluster_centers_[1],
        }

    def get_team(
        self,
        frame: np.ndarray,
        bbox: list[float],
        player_id: int,
    ) -> int:
        """Return the team ID (1 or 2) for the given player.

        The result is cached so each player is assigned to the same team
        across frames.

        Args:
            frame: The video frame as a BGR NumPy array.
            bbox: Bounding box [x1, y1, x2, y2] for the player.
            player_id: Unique identifier for the player in the current frame.

        Returns:
            Team ID: 1 or 2.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            return 1

        jersey_color = self._get_jersey_color(frame, bbox)
        team_id = int(self.kmeans.predict([jersey_color])[0]) + 1
        self.player_team_dict[player_id] = team_id
        return team_id
