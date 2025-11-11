import json
import time
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO

Point = Tuple[int, int]
NormPoint = Tuple[float, float]

def poly_contains_point(poly: np.ndarray, pt: Point) -> bool:
    x, y = map(float, pt)
    return cv2.pointPolygonTest(poly.astype(np.int32), (x, y), False) >= 0

def any_corner_in_poly(poly: np.ndarray, box: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = map(int, box)
    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    return any(poly_contains_point(poly, c) for c in corners)

def bbox_center(box: Tuple[int, int, int, int]) -> Point:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def scale_norm_points_to_pixels(points: List[NormPoint], w: int, h: int) -> np.ndarray:
    pts = np.array([(int(nx * w), int(ny * h)) for nx, ny in points], dtype=np.int32)
    return pts.reshape(-1, 1, 2)

def load_zones(json_path: Path) -> List[List[NormPoint]]:
    if not json_path.exists():
        raise FileNotFoundError(f"no file: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    zones = data.get("zones", [])
    if not zones:
        raise ValueError(f"error with file {json_path}")
    return zones

def choose_person_indices(model: YOLO) -> List[int]:
    idxs = []
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == "person":
                try:
                    idxs.append(int(k))
                except Exception:
                    pass
    if not idxs:
        idxs = [0]
    return idxs

def run_detection(video_path: str, weights: str, device: str, zones_json: Path, conf: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"error with file {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    zones_norm = load_zones(zones_json)
    model = YOLO(weights)
    person_ids = choose_person_indices(model)
    alarm_on = False
    last_inside_ts = -1.0
    alarm_delay = 3.0
    window_name = "Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        zone_polys = [scale_norm_points_to_pixels(z, w, h) for z in zones_norm]
        overlay = frame.copy()
        for poly in zone_polys:
            if len(poly) >= 3:
                cv2.polylines(overlay, [poly], True, (0, 255, 255), 2)
                cv2.fillPoly(overlay, [poly], (0, 255, 255))
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
        results = model.predict(source=frame, stream=False, verbose=False,
                                device=None if device == "auto" else device, conf=conf)
        res = results[0]
        inside_any = False
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, p in zip(xyxy, cls, confs):
                if c not in person_ids:
                    continue
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 180, 255), 2)
                label = f"person {p:.2f}"
                cv2.putText(frame, label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 2)
                box = (x1, y1, x2, y2)
                ctr = bbox_center(box)
                for poly in zone_polys:
                    if poly_contains_point(poly, ctr) or any_corner_in_poly(poly, box):
                        inside_any = True
                        cv2.circle(frame, ctr, 4, (0, 0, 255), -1)
                        break
        now = time.time()
        if inside_any:
            alarm_on = True
            last_inside_ts = now
        elif alarm_on and last_inside_ts > 0 and (now - last_inside_ts) >= alarm_delay:
            alarm_on = False
        if alarm_on:
            cv2.putText(frame, "ALARM!", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
        info = f"Device: {device} | FPS: {fps:.1f} | Zones: {len(zones_norm)} | Conf: {conf}"
        cv2.putText(frame, info, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
    cap.release()
    cv2.destroyWindow(window_name)

if __name__ == "__main__":
    VIDEO = r"2.mp4"
    WEIGHTS = r"best.pt"
    ZONES = r"zones.json"
    DEVICE = "auto"
    CONF = 0.25
    
    run_detection(VIDEO, WEIGHTS, DEVICE, Path(ZONES), CONF)
