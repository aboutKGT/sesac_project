import os
import json
import numpy as np
from pycocotools import mask as maskUtils
from typing import List, Dict, Any

# --- 1. 설정 변수 ---
LABEL_DIR = '/home/elicer/dev/detectron2/final_data/val/labels'
OUTPUT_JSON_PATH = '/home/elicer/dev/gt/data/label_data/val_coco_format.json'

# --- 2. 카테고리 정의, 정리 및 통합 ---

# 2-1. 클래스 통합 맵 정의
# 원본 데이터(Key)를 새로운 목표 클래스(Value)로 매핑합니다.
CONSOLIDATION_MAP = {
    "freespace": "Freespace",       # 원본: 소문자 -> 목표: 대문자
    "trafficSign": "TrafficSign",   # 원본: trafficSign -> 목표: TrafficSign
    "trafficLight": "TrafficLight", # 원본: trafficLight -> 목표: TrafficLight (예상)
    "fense": "background",          # 원본: fense(오타) -> 목표: background (혹은 무시)

    # Thing Class 통합
    "policeCar": "otherCar",
    "ambulance": "otherCar",
    "schoolBus": "otherCar",
    "twoWheeler": "motorcycle", # 두 바퀴는 오토바이로 통합
    
    # Stuff Class 통합
    "safetyZone": "roadMark",
    "speedBump": "roadMark",
    "blueLane": "roadMark",
    "redLane": "roadMark",
    "stoplane": "roadMark",
}

# 2-2. 완전히 무시하고 제거할 클래스 정의
IGNORE_CLASSES = ["nibberCore", "lense", "egoVehicle", "warmingTriangle", "background"]

# 2-3. 최종 Thing/Stuff 클래스 정의
THING_CLASSES = [
    "vehicle", "bus", "truck", "otherCar", # 통합된 차량
    "motorcycle", "bicycle", "pedestrian", "rider",
    "TrafficSign", "TrafficLight", "constructionGuide", "trafficDrum", 
]

STUFF_CLASSES = [
    "Freespace", "curb", "sidewalk", "crosswalk", 
    "roadMark", # 통합된 도로 구조물 및 특수 차선
    "whiteLane", "yellowLane", 
    # "background"
]

# 2-4. 최종 COCO 클래스 목록 및 맵 생성
COCO_CLASSES = THING_CLASSES + STUFF_CLASSES

CATEGORIES: List[Dict[str, Any]] = []
for i, name in enumerate(COCO_CLASSES):
    is_thing = 1 if name in THING_CLASSES else 0
    CATEGORIES.append({
        "id": i + 1, 
        "name": name, 
        "supercategory": "none", 
        "is_thing": is_thing # Panoptic FPN 학습에 필수 플래그
    })

CATEGORY_MAP = {cat['name']: cat['id'] for cat in CATEGORIES} 

# --- 3. 헬퍼 함수: Polygon을 Bbox와 정확한 Area로 변환 ---
def polygon_to_bbox_and_area(polygon_coords: List[float], image_width: int, image_height: int):
    """
    폴리곤 좌표를 COCO 포맷의 bbox, area, segmentation으로 변환합니다.
    Area는 pycocotools를 사용하여 정확한 폴리곤 픽셀 면적으로 계산
    """
    # COCO segmentation 포맷: [x1, y1, x2, y2, ...] 형태의 리스트의 리스트
    segmentation = [[float(c) for c in polygon_coords]]
    
    # Bbox 계산
    points = np.array(polygon_coords).reshape(-1, 2)
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    bbox = [float(x_min), float(y_min), float(width), float(height)] # [x_min, y_min, width, height]
    
    # Area 계산: pycocotools 사용 
    try:
        rles = maskUtils.frPyObjects(segmentation, image_height, image_width)
        area = float(maskUtils.area(rles).sum())
    except Exception:
        # 오류 발생 시 Bbox 면적으로 대체
        area = float(width * height) 
        
    return bbox, area, segmentation

# --- 4. 메인 변환 함수 ---
def convert_to_coco():
    print("COCO 포맷 변환 시작...")
    # 이미지 정보 추출 시 int 변환을 위해 width, height 변수 선언
    width, height = 0, 0 
    
    coco_output: Dict[str, Any] = {"info": {}, "licenses": [], "categories": CATEGORIES, "images": [], "annotations": []}
    image_id = 0
    annotation_id = 0
    
    # [디버깅용] 매핑되지 않아 버려지는 클래스를 확인하기 위한 세트
    skipped_classes = set()

    for filename in os.listdir(LABEL_DIR):
        if not filename.endswith('.json'): continue
        label_path = os.path.join(LABEL_DIR, filename)
        
        with open(label_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"오류: {filename} JSON 형식 오류. 건너뜀.")
                continue

        # **(1) 이미지 정보 추출 및 ID 할당**
        try:
            image_name = data['information']['filename']
            # 해상도 정보를 읽고 int로 변환 (Area 계산에 사용)
            width = int(data['information']['resolution'][0])
            height = int(data['information']['resolution'][1])
        except (KeyError, IndexError, TypeError) as e:
            print(f"오류: {filename}에서 이미지 해상도 정보 누락/형식 오류 ({e}). 건너뜀.")
            continue

        coco_output["images"].append({
            "id": image_id,
            "file_name": image_name,
            "height": height,
            "width": width,
        })
        
        # **(2) 어노테이션 정보 처리 및 클래스 통합/제거**
        annotations_list = data.get("annotations", []) 

        for ann in annotations_list:
            try:
                class_name = ann['class']
                
                # 공백 제거 (혹시 모를 " freespace " 같은 경우 방지)
                class_name = class_name.strip()

                # 2-1. 클래스 통합 적용
                if class_name in CONSOLIDATION_MAP:
                    class_name = CONSOLIDATION_MAP[class_name]
                
                # 2-2. 제거 대상 클래스인지 확인
                if class_name in IGNORE_CLASSES: continue
                
                # 2-3. 최종 정의된 클래스 목록에 없는지 확인
                if class_name not in CATEGORY_MAP:
                    # 디버깅: 어떤 클래스가 버려지는지 수집
                    skipped_classes.add(class_name)
                    continue 
                
                polygon_coords = ann['polygon']
                
                # Polygon 변환 실행 (이미지 크기를 인자로 전달)
                bbox, area, segmentation = polygon_to_bbox_and_area(
                    polygon_coords, width, height
                )
                
                # Area가 너무 작은(1픽셀 이하) 어노테이션은 제외하여 노이즈를 줄입니다.
                if area > 1.0: 
                    coco_output["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": CATEGORY_MAP[class_name],
                        "bbox": bbox,
                        "area": area,
                        "segmentation": segmentation, 
                        "iscrowd": 0, # Panoptic 학습 시에는 항상 0으로 유지
                    })
                    annotation_id += 1

            except Exception as e:
                # print(f"예외 발생: 어노테이션 처리 중 오류가 발생했습니다. ({filename}, {e})")
                continue
        
        image_id += 1

    # 5. 최종 COCO JSON 파일 저장
    print(f"\n[변환 완료]")
    print(f" - 총 이미지 수: {image_id}")
    print(f" - 총 어노테이션 수: {annotation_id}")
    
    if skipped_classes:
        print(f"\n[주의] 다음 클래스들은 매핑되지 않아 제외되었습니다 (의도한 것이 아니라면 CONSOLIDATION_MAP을 수정하세요):")
        print(f" -> {list(skipped_classes)}")

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, indent=4, ensure_ascii=False) 
    print(f"\n COCO JSON 파일이 성공적으로 저장되었습니다: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    convert_to_coco()
    
    
# --- 실행 결과 예시 ---
# [변환 완료]
#  - 총 이미지 수: 51872
#  - 총 어노테이션 수: 1533352

# [주의] 다음 클래스들은 매핑되지 않아 제외되었습니다 (의도한 것이 아니라면 CONSOLIDATION_MAP을 수정하세요):
#  -> ['eogVehicle', 'crossWalk', 'stopLane', 'rubberCone', 'sideWalk']

#  COCO JSON 파일이 성공적으로 저장되었습니다: /home/elicer/dev/gt/data/label_data/train_coco_format.json