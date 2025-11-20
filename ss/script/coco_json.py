import cv2
import numpy as np
import json
import os
import glob

######## json 파일 형식 변환 ########

label_dir = "/home/elicer/dev/ss/script/train_small_labels/"

coco_output = {
    "images": [],
    "annotations": [],
    "categories": []

}

category_dict = {}
category_id = 1
annotation_id = 1
image_id = 1

def polygon_to_bbox(polygon):
    """폴리곤 좌표 리스트를 bbox [x,y,w,h]로 변환"""
    xs = polygon[0::2]
    ys = polygon[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return [x_min, y_min, x_max-x_min, y_max-y_min]

count = 0 

for json_file in glob.glob(os.path.join(label_dir, "*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)


    file_name = data["information"]["filename"]
    width, height = data["information"]["resolution"]


    coco_output['images'].append({
        "id": image_id,
        "file_name" : file_name,
        "width" : width,
        "height" : height
    })

    for ann in data.get("annotations", []):
        class_name = ann["class"]
        
        if class_name.lower() == "background":
            continue

        polygon = ann["polygon"]

        if class_name not in category_dict:
            category_dict[class_name] = category_id
            coco_output['categories'].append({
                "id" : category_id,
                "name" : class_name,
                "supercategory" : "none"
            })
            category_id += 1

        bbox = polygon_to_bbox(polygon)

        coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_dict[class_name],
                "segmentation" : [polygon],
                "bbox": bbox,
                "area": bbox[2] * bbox[3],
                "iscrowd": 0
    })
        annotation_id += 1
    image_id += 1
    
    count += 1



new_dir = "/home/elicer/dev/ss/script"

# 최종 COCO json 저장
output_path = os.path.join(new_dir, "coco_train_annotations.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(coco_output, f, indent=4, ensure_ascii=False)

print(f"COCO format 파일 저장 완료: {output_path}")
print(f"json 파일 {count}개 > 1개의 json파일로 변경")