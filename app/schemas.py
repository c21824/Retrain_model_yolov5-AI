from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class RetrainRequest(BaseModel):
    dataset_id: int
    model_id: int
    output_path: Optional[str] = None
    create_yolo: Optional[bool] = True

class ExportResult(BaseModel):
    dataset_id: int
    dataset_path: str
    images_copied: int
    labels_written: int
    classes: List[str]
    details: Dict[str, Any]
