from pathlib import Path
from typing import List, Optional, Any

from dataclasses import dataclass


@dataclass
class AnnotationItem:
    bbox: List
    text: str
    lines: int
    image_filepath: Optional[Path] = None
    image: Any = None
