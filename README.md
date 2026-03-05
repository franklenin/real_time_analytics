# real_time_analytics

Real-time soccer video analysis using YOLO-v8.  The script detects players in
each frame, assigns them to one of two teams by jersey colour, and displays a
live count per team directly on the video.

## Features

- **Player detection** – YOLO-v8 (any model size: n / s / m / l / x)
- **Team assignment** – K-means clustering on jersey colours; no manual labelling
- **Real-time overlay** – bounding boxes and live player counts per team
- **Optional output** – save the annotated video to disk

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

## Installation

1. Clone the repository:

```bash
git clone https://github.com/franklenin/real_time_analytics.git
cd real_time_analytics
```

2. Install **uv** (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install project dependencies:

```bash
uv sync
```

## Usage

```bash
uv run main.py <video> [--output OUTPUT] [--model MODEL]
```

| Argument | Description | Default |
|---|---|---|
| `video` | Path to the input soccer video | required |
| `--output`, `-o` | Path to save the annotated output video | display only |
| `--model`, `-m` | YOLO model weights (e.g. `yolov8n.pt`, `yolov8x.pt`) | `yolov8n.pt` |

### Examples

Display the analysis in real-time:

```bash
uv run main.py soccer_game.mp4
```

Save the annotated video:

```bash
uv run main.py soccer_game.mp4 --output analyzed_game.mp4
```

Use a more accurate (but slower) model:

```bash
uv run main.py soccer_game.mp4 --model yolov8x.pt
```

Press **q** to quit the live window at any time.

## How It Works

1. **Detection** – every frame is fed through YOLO-v8, which returns bounding
   boxes for each person with confidence ≥ 0.5.
2. **Jersey colour extraction** – the top half of each bounding box (the jersey
   area) is analysed with a two-cluster K-means model; the non-background
   cluster is taken as the player's jersey colour.
3. **Team clustering** – on the first frame that contains at least two players,
   all jersey colours are clustered into two groups (Team 1 / Team 2) using
   K-means.  The same cluster model is reused for all subsequent frames.
4. **Annotation** – bounding boxes are drawn in the team colour and a
   scoreboard with per-team player counts is rendered in the top-left corner.