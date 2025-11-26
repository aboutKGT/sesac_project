import os
import cv2
import random
import glob
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# --- [사용자 설정] ---
# 1. 학습 결과가 저장된 폴더 경로 (model_final.pth와 config.yaml이 있는 곳)
# 예: '/home/elicer/dev/gt/custom_model/20231119_120000' (실제 폴더명으로 수정하세요)
MODEL_DIR = '/home/elicer/dev/gt/custom_model/20251119_090609' 

# 2. 추론할 이미지가 있는 폴더 (Validation Images)
VAL_IMAGE_ROOT = '/home/elicer/dev/detectron2/final_data/val/images'

# 3. 결과 이미지를 저장할 폴더
SAVE_DIR = os.path.join(MODEL_DIR, 'inference_results_val_100')

# 4. 클래스 정의 (학습 코드와 정확히 동일해야 함)
THING_CLASSES = [
    "vehicle", "bus", "truck", "otherCar",
    "motorcycle", "bicycle", "pedestrian", "rider",
    "TrafficSign", "TrafficLight", "constructionGuide", "trafficDrum", 
]
STUFF_CLASSES = [
    "Freespace", "curb", "sidewalk", "crosswalk", 
    "roadMark", "whiteLane", "yellowLane",
]
# 학습 때 이 순서로 합쳤으므로, 추론 때도 동일 순서 유지 필수
ALL_CLASSES = THING_CLASSES + STUFF_CLASSES


def setup_inference_cfg(model_dir):
    # 1. 기본 Config 생성
    cfg = get_cfg()
    
    # 2. 학습 때 저장한 config.yaml 불러오기 (모델 구조 동기화)
    config_path = os.path.join(model_dir, "config.yaml")
    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
    else:
        # 만약 config.yaml이 없다면 기본 구조 로드 (가능한 config.yaml 사용)
        from detectron2 import model_zoo
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ALL_CLASSES)

    # 3. 학습된 가중치 로드
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, "model_final.pth")
    
    # 4. 테스트(추론) 설정
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 50% 이상 확신하는 것만 표시
    cfg.DATASETS.TEST = ("my_val_dataset", )     # (더미 이름)
    
    return cfg

def main():
    # 결과 저장 디렉토리 생성
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Creating output directory: {SAVE_DIR}")

    # Config 및 Predictor 설정
    print(f"Loading model from: {MODEL_DIR}")
    if not os.path.exists(os.path.join(MODEL_DIR, "model_final.pth")):
        print("Error: model_final.pth not found!")
        return

    cfg = setup_inference_cfg(MODEL_DIR)
    predictor = DefaultPredictor(cfg)

    # 시각화를 위한 메타데이터 수동 등록
    # (학습 데이터셋을 다시 등록하지 않고, 클래스 이름만 메타데이터에 주입)
    metadata = MetadataCatalog.get("my_val_inference")
    metadata.thing_classes = ALL_CLASSES
    
    # 이미지 파일 리스트 가져오기
    # jpg, png 등 확장자 고려
    image_extensions = ["*.jpg", "*.jpeg", "*.png"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(VAL_IMAGE_ROOT, ext)))
    
    if not image_paths:
        print(f"No images found in {VAL_IMAGE_ROOT}")
        return

    # 랜덤 샘플링 (최대 100장, 이미지가 적으면 전체 사용)
    num_samples = min(100, len(image_paths))
    sampled_images = random.sample(image_paths, num_samples)
    
    print(f"Starting inference on {num_samples} images...")

    for i, img_path in enumerate(sampled_images):
        # 이미지 읽기
        im = cv2.imread(img_path)
        if im is None:
            continue
            
        # 추론 수행
        outputs = predictor(im)
        
        # 시각화
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.8, instance_mode=ColorMode.IMAGE)
        
        # CPU로 이동 후 그리기
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # 결과 저장
        file_name = os.path.basename(img_path)
        save_path = os.path.join(SAVE_DIR, f"result_{file_name}")
        
        # opencv는 BGR 형식이므로 RGB 변환된 visualizer 출력을 다시 BGR로 변환하여 저장
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_samples} images...")

    print(f"\n[완료] 모든 결과 이미지가 저장되었습니다: {SAVE_DIR}")

if __name__ == "__main__":
    main()