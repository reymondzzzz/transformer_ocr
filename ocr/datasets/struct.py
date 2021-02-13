from pathlib import Path
from typing import List

from dataclasses import dataclass


@dataclass
class AnnotationItem:
    image_filepath: Path
    bbox: List
    text: str
    lines: int
