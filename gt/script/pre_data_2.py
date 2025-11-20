import os
import json
import glob
import numpy as np
import cv2
from datetime import datetime
from pycocotools import mask as maskUtils

# =========================================================
# [설정] 경로 설정
# =========================================================
INPUT_DIR = '/home/elicer/dev/gt/data/new_data/label'  # Superb AI json 파일들이 있는 폴더
OUTPUT_JSON_PATH = '/home/elicer/dev/gt/data/label_data/more_data_coco.json'

# =========================================================
# [클래스 매핑] Superb AI의 라벨 이름 -> 우리의 학습용 소문자 클래스
# =========================================================
CLASS_MAPPING = {
    # [차량 관련]
    "Car": "vehicle",
    "car-b": "vehicle",             # 박스용 라벨도 통합
    "TruckBus": "truck",            # 트럭/버스는 일단 truck으로 (bus로 분리 원하면 수정)
    "TruckBus-b": "truck",
    "Two-wheel Vehicle": "motorcycle",
    "Two-wheel Vehicle-b": "motorcycle",
    "Personal Mobility": "motorcycle", # 킥보드 등도 오토바이류로 통합
    
    # [사람 관련]
    "Adult": "pedestrian",
    "Kid student": "pedestrian",
    "Pedestrian-b": "pedestrian",
    
    # [도로/시설물]
    "Traffic Sign": "trafficsign",
    "Traffic Light": "trafficlight",
    "Speed bump": "roadmark",
    "Parking space": "roadmark",
    "Crosswalk": "crosswalk",
    
    # 무시할 것들
    "b": "ignore" # 혹시 모를 더미
}

# 최종 클래스 정의 (순서 중요)
THING_CLASSES = [
  "vehicle", "bus", "truck", "othercar",
  "motorcycle", "bicycle", "pedestrian", "rider",
  "trafficsign", "trafficlight", "constructionguide", "trafficdrum", 
]
STUFF_CLASSES = [
  "freespace", "curb", "sidewalk", "crosswalk", 
  "roadmark", "whitelane", "yellowlane", 
]
ALL_CLASSES = THING_CLASSES + STUFF_CLASSES
CATEGORY_MAP = {name: i+1 for i, name in enumerate(ALL_CLASSES)}

def rle_to_polygon(segmentation):
    """
    RLE(Run-Length Encoding)를 Polygon(점들의 좌표)으로 변환
    """
    # 1. RLE 디코딩 -> 바이너리 마스크 (0과 1의 이미지)
    try:
        mask = maskUtils.decode(segmentation)
    except:
        return None, 0

    # 2. 마스크에서 외곽선(Contour) 추출
    # cv2.CHAIN_APPROX_SIMPLE: 불필요한 점을 줄여서 용량 최적화
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    total_area = 0
    
    for contour in contours:
        if len(contour) < 3: continue # 점 3개 미만은 면적 없음
        
        # [x, y] 형태로 변환 및 1차원으로 펼치기 (COCO 포맷)
        contour = contour.flatten().tolist()
        
        # 면적이 너무 작은 노이즈 제거 (예: 10픽셀 미만)
        if len(contour) > 6: # 최소 x,y,x,y,x,y
            polygons.append(contour)
            
    # 면적 계산 (마스크 기준)
    total_area = float(maskUtils.area(segmentation))
    
    return polygons, total_area

def convert_superb_rle_to_coco():
    print("Superb AI (RLE) -> COCO (Polygon) 변환 시작...")
    
    coco_output = {
        "info": {"description": "Converted from Superb AI RLE", "date_created": datetime.now().isoformat()},
        "licenses": [],
        "categories": [{"id": v, "name": k} for k, v in CATEGORY_MAP.items()],
        "images": [],
        "annotations": []
    }

    image_id_cnt = 0
    annotation_id_cnt = 0
    
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    print(f"총 {len(json_files)}개의 파일을 처리합니다.")

    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 1. 파일 내부 ID -> 이름 매핑 테이블 생성
        # Superb AI는 파일마다 category id 매핑이 다를 수 있으므로 매번 생성
        local_id_to_name = {}
        if 'category' in data:
            for cat in data['category']:
                local_id_to_name[cat['id']] = cat['name']
        
        # 2. 이미지 정보 처리
        # images 리스트가 있거나, 파일명으로 유추
        file_name = os.path.basename(json_file).replace('.json', '') # 일단 확장자 떼고
        # 확장자가 jpg인지 png인지 확인 (파일이 있다고 가정)
        # Superb AI json 안에 보통 'images' -> 'file_name'에 전체 경로가 있음
        if data.get('images'):
            img_meta = data['images'][0]
            file_name = os.path.basename(img_meta.get('file_name', file_name + '.png'))
            height = img_meta.get('height', 1080)
            width = img_meta.get('width', 1920)
        else:
            file_name += ".png" # 기본값
            height, width = 1080, 1920

        coco_output['images'].append({
            "id": image_id_cnt,
            "file_name": file_name,
            "height": height,
            "width": width,
            "license": 0,
            "date_captured": ""
        })

        # 3. 어노테이션 변환
        for ann in data.get('annotations', []):
            # (1) 클래스 이름 찾기
            cat_id = ann.get('category_id')
            raw_name = local_id_to_name.get(cat_id, "unknown")
            
            # (2) 타겟 클래스로 매핑
            target_class = CLASS_MAPPING.get(raw_name)
            if not target_class or target_class not in CATEGORY_MAP:
                continue
                
            target_cat_id = CATEGORY_MAP[target_class]
            
            # (3) RLE -> Polygon 변환 (핵심!)
            # 박스만 있고 segmentation이 없는 경우도 처리
            segmentation = ann.get('segmentation', None)
            polygons = []
            area = ann.get('area', 0)
            
            if segmentation and isinstance(segmentation, dict) and 'counts' in segmentation:
                polygons, area = rle_to_polygon(segmentation)
                if not polygons: continue # 변환 실패 혹은 빈 마스크
            else:
                # 세그멘테이션 정보가 없으면 박스만이라도 살릴지 결정
                # 여기서는 스킵 (Instance Segmentation 학습이므로)
                continue

            # (4) COCO Annotation 추가
            coco_output['annotations'].append({
                "id": annotation_id_cnt,
                "image_id": image_id_cnt,
                "category_id": target_cat_id,
                "bbox": ann.get('bbox', [0,0,0,0]),
                "area": area,
                "segmentation": polygons,
                "iscrowd": 0
            })
            annotation_id_cnt += 1
            
        image_id_cnt += 1
        
    # 저장
    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)
    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, ensure_ascii=False, indent=4)
        
    print(f"\n변환 완료: {OUTPUT_JSON_PATH}")
    print(f" - Images: {image_id_cnt}")
    print(f" - Annotations: {annotation_id_cnt}")

if __name__ == "__main__":
    convert_superb_rle_to_coco()