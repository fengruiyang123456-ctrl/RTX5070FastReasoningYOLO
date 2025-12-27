from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


COCO80_NAMES: List[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class AppConfig:
    root_dir: Path = Path(__file__).resolve().parents[2]
    weights_dir: Path = root_dir / "weights"
    outputs_dir: Path = root_dir / "outputs"

    input_width: int = 640
    input_height: int = 640
    conf_thres: float = 0.25
    iou_thres: float = 0.45
    device: str = "cuda:0"

    class_names: Optional[List[str]] = None

    def get_class_names(self) -> List[str]:
        return self.class_names or COCO80_NAMES

    def weights_path(self, name: str) -> Path:
        return self.weights_dir / name
