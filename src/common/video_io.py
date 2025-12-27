from typing import Generator, Optional, Tuple, Union

import cv2


def open_video(source: Union[str, int]) -> cv2.VideoCapture:
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")
    return cap


def iter_frames(
    cap: cv2.VideoCapture,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[bool, Optional[cv2.Mat]], None, None]:
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield ok, frame
        count += 1
        if max_frames is not None and count >= max_frames:
            break
