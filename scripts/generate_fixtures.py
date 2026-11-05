"""Generate the synthetic test fixtures in tests/fixtures/ (simple shapes, no real footage)."""

from pathlib import Path

import cv2
import numpy as np

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
WIDTH, HEIGHT = 320, 240
VIDEO_FPS = 15
VIDEO_FRAMES = 45
BACKGROUND = (60, 60, 60)
BOX_COLOR = (40, 180, 240)
BOX_SIZE = 60


def draw_scene(box_x: int) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    top = (HEIGHT - BOX_SIZE) // 2
    cv2.rectangle(image, (box_x, top), (box_x + BOX_SIZE, top + BOX_SIZE), BOX_COLOR, thickness=-1)
    return image


def write_video(path: Path) -> None:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, VIDEO_FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {path}")
    for index in range(VIDEO_FRAMES):
        writer.write(draw_scene(index * (WIDTH - BOX_SIZE) // (VIDEO_FRAMES - 1)))
    writer.release()


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    write_video(FIXTURES_DIR / "sample.mp4")
    cv2.imwrite(str(FIXTURES_DIR / "sample.png"), draw_scene((WIDTH - BOX_SIZE) // 2))
    print(f"Fixtures written to {FIXTURES_DIR}")


if __name__ == "__main__":
    main()
