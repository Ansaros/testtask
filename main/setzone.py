import json
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

Point = Tuple[int, int]
NormPoint = Tuple[float, float]


def scale_pixels_to_norm(points: List[Point], w: int, h: int) -> List[NormPoint]:
    return [(max(0, min(1, x / (w - 1))), max(0, min(1, y / (h - 1)))) for x, y in points]

def save_zones(json_path: Path, zones: List[List[NormPoint]]):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"zones": zones}, f, ensure_ascii=False, indent=2)
    print(f"saved to {json_path}")

class ZoneMarker:
    """
        Left Click: add point,
        Z: finalize polygon,
        R: remove last polygon,
        C: clear,
        S: save",
        Q / ESC: exit,
    """
    def __init__(self, window: str, frame: np.ndarray):
        self.window = window
        self.h, self.w = frame.shape[:2]
        self.polygons: List[List[Point]] = []
        self.current: List[Point] = []
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current.append((x, y))

    def draw(self, base: np.ndarray) -> np.ndarray:
        img = base.copy()
        overlay = img.copy()
        for poly in self.polygons:
            if len(poly) >= 3:
                cv2.polylines(overlay, [np.array(poly, dtype=np.int32)], True, (0, 255, 255), 2)
                cv2.fillPoly(overlay, [np.array(poly, dtype=np.int32)], (0, 255, 255))
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        if len(self.current) >= 2:
            cv2.polylines(img, [np.array(self.current, dtype=np.int32)], False, (0, 165, 255), 2)
        for p in self.current:
            cv2.circle(img, p, 3, (0, 140, 255), -1)
        return img

    def finalize_current(self):
        if len(self.current) >= 3:
            self.polygons.append(self.current.copy())
        self.current = []

    def to_normalized(self) -> List[List[NormPoint]]:
        all_polys = self.polygons.copy()
        if len(self.current) >= 3:
            all_polys.append(self.current)
        return [scale_pixels_to_norm(poly, self.w, self.h) for poly in all_polys]


def mark_zones(video_path: str, out_json: Path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("failed to read")

    window_name = "Marker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    marker = ZoneMarker(window_name, frame)

    help_lines = [
        "Left Click: add point",
        "Z: finalize polygon",
        "R: remove last polygon",
        "C: clear",
        "S: save",
        "Q / ESC: exit",
    ]

    while True:
        img = marker.draw(frame)
        y = 25
        for hl in help_lines:
            cv2.putText(img, hl, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 25
        cv2.imshow(window_name, img)

        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("z"):
            marker.finalize_current()
        elif key == ord("r"):
            if marker.polygons:
                marker.polygons.pop()
        elif key == ord("c"):
            marker.polygons.clear()
            marker.current.clear()
        elif key == ord("s"):
            zones_norm = marker.to_normalized()
            save_zones(out_json, zones_norm)
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    VIDEO = r"test.mp4"
    ZONES = r"zones.json"
    mark_zones(VIDEO, Path(ZONES))
