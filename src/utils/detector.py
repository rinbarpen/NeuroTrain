from ultralytics import YOLO
import numpy as np
from PIL import Image
from pathlib import Path

class Detector:
    def __init__(self, model_name="yolov8x.pt", device="cuda"):
        self.model = YOLO(model_name)
        self.device = device
        self.model.to(self.device)
        self.model.eval()
    
    def detect(self, image: np.ndarray | Image.Image | Path | str):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image, mode="RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")
        elif isinstance(image, Path):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, str):
            image = Image.open(image).convert("RGB")

        width, height = image.size
        results = self.model(image, verbose=False)
        bbox_list = []
        for box in results[0].boxes:
            # YOLO输出的是xyxy格式，归一化到0-1
            coords = box.xyxy.cpu().numpy()[0]
            x1, y1, x2, y2 = coords[:4]
            x1_norm = x1 / width
            y1_norm = y1 / height
            x2_norm = x2 / width
            y2_norm = y2 / height
            conf = float(box.conf.cpu().numpy())
            cls = int(box.cls.cpu().numpy())
            bbox_list.append({
                "bbox": [x1_norm, y1_norm, x2_norm, y2_norm],
                "conf": conf,
                "cls": cls
            })
        return bbox_list

if __name__ == "__main__":
    detector = Detector()
    bbox_list = detector.detect("data/coco/val2017/000000000139.jpg")
    from pprint import pp
    pp(bbox_list)

    val_dir = Path("data/coco/val2017")
    for img_path in val_dir.glob("*.jpg"):
        bbox_list = detector.detect(img_path)
        result = {
            'bbox': bbox_list,
            'image_id': img_path.stem,
            'image_path': str(img_path),
        }
        import json
        with open('data/coco/val2017_bbox.json', "w") as f:
            json.dump(result, f, indent=2)
    train_dir = Path("data/coco/train2017")
    for img_path in train_dir.glob("*.jpg"):
        bbox_list = detector.detect(img_path)
        result = {
            'bbox': bbox_list,
            'image_id': img_path.stem,
            'image_path': str(img_path),
        }
        import json
        with open('data/coco/train2017_bbox.json', "w") as f:
            json.dump(result, f, indent=2)
