import os
import json
import cv2
import numpy as np
import random
import glob
from pycocotools import mask as maskUtils

# =========================================================
# [설정] 경로 설정
TARGET_DIR = '/home/elicer/dev/gt/data/test_data'
OUTPUT_DIR = '/home/elicer/dev/gt/data/test_data_vis_rle'
# =========================================================

def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

def visualize_rle_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    json_files = glob.glob(os.path.join(TARGET_DIR, "*.json"))
    
    print(f"총 {len(json_files)}개의 파일 확인됨. RLE 시각화를 시작합니다...")

    for json_path in json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 1. 카테고리 매핑 (ID -> Name)
            # Superb AI 포맷은 'category' 리스트에 이름 정보가 있음
            cat_map = {}
            if 'category' in data:
                for cat in data['category']:
                    cat_map[cat['id']] = cat['name']
            
            # 2. 이미지 찾기
            json_filename = os.path.basename(json_path)
            file_base = os.path.splitext(json_filename)[0]
            
            # 이미지 후보 찾기
            img_candidates = [
                file_base + ".png", file_base + ".jpg",
                data['images'][0]['file_name'] if data.get('images') else ""
            ]
            
            image_path = None
            for cand in img_candidates:
                if not cand: continue
                cand_name = os.path.basename(cand)
                full_path = os.path.join(TARGET_DIR, cand_name)
                if os.path.exists(full_path):
                    image_path = full_path
                    break
            
            if not image_path:
                print(f"[Skip] 이미지 없음: {json_filename}")
                continue
                
            image = cv2.imread(image_path)
            if image is None: continue
            
            # 마스크를 그릴 오버레이 레이어
            overlay = image.copy()
            
            # 3. 어노테이션 순회 (RLE 디코딩)
            annotations = data.get('annotations', [])
            
            for ann in annotations:
                cat_id = ann.get('category_id')
                label_name = cat_map.get(cat_id, str(cat_id))
                
                # RLE 데이터 확인
                seg = ann.get('segmentation')
                if not seg or not isinstance(seg, dict) or 'counts' not in seg:
                    continue
                
                # (핵심) RLE 문자열을 바이너리 마스크(0, 1)로 변환
                # pycocotools가 압축된 counts 문자열을 풀어서 numpy 배열로 줍니다.
                # 이미지 크기와 마스크 크기가 맞아야 함
                h, w = image.shape[:2]
                
                # 만약 segmentation size와 이미지 size가 다르면 주의 필요
                # 여기서는 pycocotools가 알아서 처리하도록 넘김
                mask = maskUtils.decode(seg)
                
                # 마스크가 있으면 색칠하기
                if mask is not None:
                    color = get_random_color()
                    
                    # 마스크 영역(값이 1인 곳)에 색상 적용
                    # overlay[mask == 1] = color  <-- 이렇게 하면 너무 진하게 덮어버림
                    
                    # OpenCV로 마스크 영역만 추출해서 색 입히기
                    mask_bool = mask.astype(bool)
                    overlay[mask_bool] = color
                    
                    # 외곽선(Contour) 그려서 구분감 주기
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image, contours, -1, (255, 255, 255), 1)

                    # 텍스트 표시 (Bbox 기준)
                    bbox = ann.get('bbox', [])
                    if bbox:
                        x, y, w, h = map(int, bbox)
                        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(image, label_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # 4. 합치기 (투명도 적용)
            final_image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
            
            save_name = f"vis_rle_{os.path.basename(image_path)}"
            save_path = os.path.join(OUTPUT_DIR, save_name)
            cv2.imwrite(save_path, final_image)
            print(f" - 저장 완료: {save_name}")

        except Exception as e:
            print(f"[Error] {json_filename}: {e}")

    print(f"\n확인 경로: {OUTPUT_DIR}")

if __name__ == "__main__":
    visualize_rle_dataset()