import json
import os
import random
from collections import defaultdict

# --- [설정] 입력 경로 ---
# 1. 고속도로 검증셋 (거리 정보 O)
HWAY_VAL_JSON = '/home/elicer/data/090_AD_Hway_Day/Validation/02_label_data/train_coco_with_distance.json'
HWAY_VAL_ROOT = '/home/elicer/data/090_AD_Hway_Day/Validation/02_label_data/2D'

# 2. 도심 검증셋 (거리 정보 X)
CITY_VAL_JSON = '/home/elicer/data/092_AD_City_Day/Validation/02_label_data/lable_day_clear/city_data_coco.json'
CITY_VAL_ROOT = '/home/elicer/data/092_AD_City_Day/Validation/01_raw_data/image_day_clear'

# --- [설정] 출력 경로 ---
OUTPUT_JSON = '/home/elicer/data/090_AD_Hway_Day/Validation/mini_val_stratified.json'

# --- [설정] 추출 목표 ---
IMAGES_PER_CLASS = 10   # 클래스당 최소 10장
IMAGES_FOR_DISTANCE = 20 # 거리 정보가 있는 이미지 최소 20장 (넉넉하게)

ALL_CLASSES = [
  "vehicle", "bus", "truck", "othercar",
  "motorcycle", "bicycle", "pedestrian", "rider",
  "trafficsign", "trafficlight", "constructionguide", "trafficdrum", 
  "freespace", "curb", "sidewalk", "crosswalk", 
  "roadmark", "whitelane", "yellowlane"
]

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_mini_validation():
    print(">>> Mini Validation 데이터셋 생성을 시작합니다...")
    
    hway_data = load_json(HWAY_VAL_JSON)
    city_data = load_json(CITY_VAL_JSON)
    
    # 데이터셋별 샘플 저장소
    # 구조: { class_name: [ {'type': 'hway', 'image': img, 'anns': [...]}, ... ] }
    class_to_samples = defaultdict(list)
    distance_samples = [] 

    def parse_and_collect(dataset, dataset_type, root_dir):
        images = {img['id']: img for img in dataset['images']}
        img_to_anns = defaultdict(list)
        for ann in dataset['annotations']:
            img_to_anns[ann['image_id']].append(ann)
            
        # 카테고리 ID -> 이름 매핑
        cat_id_to_name = {c['id']: c['name'] for c in dataset['categories']}

        for img_id, img_info in images.items():
            # [중요] 나중에 경로 문제를 피하기 위해 절대 경로로 변환
            img_info = img_info.copy()
            img_info['file_name'] = os.path.join(root_dir, img_info['file_name'])
            
            anns = img_to_anns[img_id]
            if not anns: continue

            # 이 이미지에 포함된 클래스 파악
            classes_in_img = set()
            has_valid_distance = False
            
            for ann in anns:
                c_name = cat_id_to_name.get(ann['category_id'])
                if c_name in ALL_CLASSES:
                    classes_in_img.add(c_name)
                
                # 거리 정보 체크 (고속도로 데이터만 해당)
                if dataset_type == 'hway' and ann.get('distance', -1) != -1:
                    has_valid_distance = True
            
            sample_item = {'type': dataset_type, 'image': img_info, 'annotations': anns}
            
            # 클래스별 리스트에 추가
            for c_name in classes_in_img:
                class_to_samples[c_name].append(sample_item)
                
            # 거리 샘플 리스트에 추가
            if has_valid_distance:
                distance_samples.append(sample_item)

    print(" 1. 고속도로 데이터 파싱 중...")
    parse_and_collect(hway_data, 'hway', HWAY_VAL_ROOT)
    
    print(" 2. 도심 데이터 파싱 중...")
    parse_and_collect(city_data, 'city', CITY_VAL_ROOT)

    # --- 샘플링 (중복 제거를 위해 dict 사용) ---
    final_selection = {} # Key: (dataset_type, original_img_id)

    print(f"\n 3. 클래스별 균등 추출 (목표: {IMAGES_PER_CLASS}장/클래스)")
    for cls_name in ALL_CLASSES:
        candidates = class_to_samples[cls_name]
        if not candidates:
            print(f"  [Warning] '{cls_name}' 클래스 데이터가 없습니다.")
            continue
            
        k = min(len(candidates), IMAGES_PER_CLASS)
        selected = random.sample(candidates, k)
        
        for item in selected:
            key = (item['type'], item['image']['id'])
            final_selection[key] = item
            
    print(f"\n 4. 거리 데이터 추가 추출 (목표: {IMAGES_FOR_DISTANCE}장)")
    # 이미 선택된 것 제외하고 추가
    dist_candidates = [item for item in distance_samples if (item['type'], item['image']['id']) not in final_selection]
    
    if dist_candidates:
        k = min(len(dist_candidates), IMAGES_FOR_DISTANCE)
        selected_dist = random.sample(dist_candidates, k)
        for item in selected_dist:
            key = (item['type'], item['image']['id'])
            final_selection[key] = item
        print(f"  -> {k}장 추가됨.")
    else:
        print("  -> 이미 충분한 거리 데이터가 확보되었습니다.")

    # --- 최종 JSON 생성 ---
    print("\n 5. JSON 병합 및 ID 재발급")
    output_data = {
        "info": hway_data.get('info', {}),
        "licenses": [],
        "categories": hway_data['categories'], # 카테고리는 HWAY 기준 (두 데이터셋이 동일하다고 가정)
        "images": [],
        "annotations": []
    }

    new_img_id = 0
    new_ann_id = 0
    
    items = list(final_selection.values())
    random.shuffle(items) # 순서 섞기
    
    print(f"  -> 총 {len(items)}장의 이미지가 Mini Validation으로 구성됩니다.")

    for item in items:
        # 이미지 정보 등록
        img = item['image'].copy()
        img['id'] = new_img_id
        output_data['images'].append(img)
        
        # 어노테이션 정보 등록
        for ann in item['annotations']:
            a = ann.copy()
            a['id'] = new_ann_id
            a['image_id'] = new_img_id
            output_data['annotations'].append(a)
            new_ann_id += 1
            
        new_img_id += 1

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n[완료] 생성된 파일: {OUTPUT_JSON}")

if __name__ == "__main__":
    create_mini_validation()