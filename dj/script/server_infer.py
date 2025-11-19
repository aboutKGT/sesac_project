import os
import cv2
import datetime
import numpy as np
import random
import torch # 텐서 기반 필터링을 위해 추가
from typing import List, Optional

# Detectron2 관련 모듈
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# --- 1. 설정 변수 (사용자 환경에 맞게 수정 필수!) ---

# 학습 결과물이 저장된 디렉토리 경로
TRAINING_OUTPUT_DIR = "/home/elicer/dev/gt/custom_model/20251118_094633" 

# 추론할 이미지들이 있는 폴더 경로 (INPUT_IMAGE_PATH가 폴더 경로로 사용됩니다)
INPUT_IMAGE_FOLDER_PATH = "/home/elicer/dev/detectron2/final_data/val/images"

# 추론 결과를 저장할 기본 경로 (하위에 이미지별 폴더가 생성됩니다)
OUTPUT_BASE_DIR = "/home/elicer/dev/gt/output/inference_results"

# 무작위로 추론할 이미지의 개수 (N)
N_IMAGES_TO_INFER = 10 # 원하는 이미지 개수로 설정하세요.

# --- 2. 카테고리 정의 (학습 시 사용한 THING_CLASSES와 동일해야 함) ---
ALL_CLASSES = [
    "vehicle", "bus", "truck", "policeCar", "ambulance", "schoolBus", "otherCar",
    "motorcycle", "bicycle", "twoWheeler", "pedestrian", "rider",
    "Freespace", "curb", "sidewalk", "crosswalk", "safetyZone", "speedBump", 
    "roadMark", "whiteLane", "yellowLane", "blueLane", "redLane", "stoplane",
    "TrafficSign", "TrafficLight", "constructionGuide", "trafficDrum", 
    "nibberCore", "warmingTriangle", "lense", "egoVehicle", "background"
]
THING_CLASSES = ALL_CLASSES 
# -----------------------------------------------------------------


def setup_predictor(cfg_path, model_weights_path):
    """Detectron2 모델 예측기(Predictor)와 메타데이터를 설정하고 반환합니다."""
    # 1. 설정 (Config) 로드
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    
    # 2. 학습된 가중치(Weights) 경로 지정
    cfg.MODEL.WEIGHTS = model_weights_path
    
    # 3. 추론 시 사용할 최소 임계값 설정
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    
    # 4. NMS 임계값 설정
    cfg.MODEL.ROI_HEADS.NMS_THRESH = 0.5 
    
    # 5. MetadataCatalog에 클래스 정보 등록 (시각화를 위해 필수)
    metadata_name = "custom_inference"
    # 이미 등록되어 있을 수 있으므로 .get().set()을 사용합니다.
    MetadataCatalog.get(metadata_name).thing_classes = THING_CLASSES
    
    # 6. DefaultPredictor 초기화
    predictor = DefaultPredictor(cfg)
    
    return predictor, MetadataCatalog.get(metadata_name)

def load_random_images(folder_path: str, n_images: int) -> Optional[List[cv2.typing.MatLike]]:
    """
    주어진 폴더 경로에서 지정된 개수만큼 이미지를 무작위로 로드합니다.
    Args:
        folder_path (str): 이미지가 포함된 폴더의 경로.
        n_images (int): 무작위로 로드할 이미지의 개수.
    Returns:
        Optional[List[cv2.typing.MatLike]]: 성공적으로 로드된 이미지(NumPy 배열) 리스트.
                                             오류 발생 시 None을 반환합니다.
    """
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    all_image_paths = []

    # 1. 유효성 검사 및 경로 목록 생성
    if not os.path.isdir(folder_path):
        print(f"오류: 지정된 경로 '{folder_path}'는 유효한 폴더가 아닙니다.")
        return None
        
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_extensions):
                full_path = os.path.join(folder_path, filename)
                all_image_paths.append(full_path)
    except Exception as e:
        print(f"오류: 폴더 내용을 읽는 중 문제가 발생했습니다. {e}")
        return None

    # 2. 무작위 샘플 선택
    if not all_image_paths:
        print("경고: 폴더에 로드할 수 있는 이미지 파일이 없습니다.")
        return []

    num_available = len(all_image_paths)
    if n_images <= 0:
        print("경고: 로드할 이미지 개수(n)는 1 이상이어야 합니다.")
        return []
    
    if n_images > num_available:
        print(f"경고: 요청한 {n_images}장이 전체 {num_available}장보다 많습니다. 전체 이미지를 로드합니다.")
        n_images = num_available

    # 무작위로 n_images 개수만큼 경로 선택 (중복 없음)
    random_paths = random.sample(all_image_paths, n_images)

    # 3. 반복문으로 이미지 로드
    loaded_images = []
    print(f"총 {n_images}장의 이미지를 무작위로 로드합니다...")
    
    for i, image_path in enumerate(random_paths):
        # 한글 경로 문제 방지를 위해 imdecode 사용
        img_array = np.fromfile(image_path, np.uint8)
        im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if im is not None:
            loaded_images.append(im)
        else:
            print(f"[{i+1}/{n_images}] 로드 실패: {os.path.basename(image_path)}")

    print("--- 이미지 로드 완료 ---")
    return loaded_images

def run_inference(folder_path: str, n_images: int, output_base_dir: str):
    """
    무작위로 n장의 이미지를 로드하여 모델 추론을 실행하고 결과를 저장합니다.
    
    Args:
        folder_path (str): 이미지가 있는 폴더 경로.
        n_images (int): 추론할 무작위 이미지 개수.
        output_base_dir (str): 결과 이미지를 저장할 최상위 폴더 경로.
    """
    
    # 1. 모델 및 메타데이터 설정
    cfg_file = os.path.join(TRAINING_OUTPUT_DIR, "config.yaml")
    weights_file = os.path.join(TRAINING_OUTPUT_DIR, "model_final.pth")

    predictor, metadata = setup_predictor(cfg_file, weights_file)
    print(f"모델 로드 완료: {weights_file}")
    
    # 2. 이미지 로드 (n_images 개수만큼 로드)
    # load_random_images는 NumPy 배열(이미지 데이터)의 리스트를 반환합니다.
    loaded_images = load_random_images(folder_path, n_images)
    
    if not loaded_images:
        print(f"오류: 로드된 이미지가 없으므로 추론을 실행할 수 없습니다. 경로를 확인해주세요: {folder_path}")
        return

    # 결과 저장 디렉토리 생성 (타임스탬프를 추가하여 덮어쓰기 방지)
    thistime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_base_dir, f"inference_batch_{thistime}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"결과 이미지는 다음 폴더에 저장됩니다: {output_dir}")

    # 3. 로드된 각 이미지에 대해 추론 실행
    for i, im in enumerate(loaded_images):
        
        # 3-1. 추론 실행
        outputs = predictor(im)
        
        # 3-2. 클래스 필터링 (원하는 클래스만 시각화)
        # 현재는 ALL_CLASSES를 그대로 사용하므로 필터링은 생략되지만,
        # 특정 클래스만 보고 싶을 때를 대비해 로직은 유지합니다.
        DESIRED_CLASSES = ["vehicle", "pedestrian", "Freespace"] 
        
        # [
        #     "vehicle", "bus", "truck", "policeCar", "ambulance", "schoolBus", "otherCar",
        #     "motorcycle", "bicycle", "twoWheeler", "pedestrian", "rider",
        #     "Freespace", "curb", "sidewalk", "crosswalk", "safetyZone", "speedBump", 
        #     "roadMark", "whiteLane", "yellowLane", "blueLane", "redLane", "stoplane",
        #     "TrafficSign", "TrafficLight", "constructionGuide", "trafficDrum", 
        #     "nibberCore", "warmingTriangle", "lense", "egoVehicle", "background"
        # ]
        
        instances = outputs["instances"].to("cpu")
        
        if DESIRED_CLASSES != metadata.thing_classes:
            # 원하는 클래스 ID 리스트를 PyTorch 텐서로 생성
            desired_ids = torch.tensor(
                [metadata.thing_classes.index(cls_name) for cls_name in DESIRED_CLASSES],
                dtype=torch.long
            )
            
            # 예측된 클래스 ID를 가져옵니다.
            pred_classes = instances.pred_classes
            
            # torch.isin을 사용하여 필터 마스크를 생성하고 인스턴스를 필터링합니다.
            filter_mask = torch.isin(pred_classes, desired_ids)
            instances = instances[filter_mask]
        
        # 필터링된 인스턴스를 다시 outputs에 할당
        outputs["instances"] = instances
        
        # 4. 결과 시각화
        # Visualizer는 RGB 입력을 기대하므로 BGR인 im을 변환하여 전달
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)
        
        out = v.draw_instance_predictions(outputs["instances"])
        
        # 5. 결과 저장 (각 이미지마다 고유한 파일명 사용)
        # RGB 결과를 다시 BGR로 변환하여 OpenCV로 저장
        result_img = out.get_image()[:, :, ::-1]
        
        output_filename = f"result_image_{i+1:02d}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        
        cv2.imwrite(output_path, result_img)
        
        print(f"✅ 이미지 {i+1}/{n_images} 추론 완료. 저장 위치: {output_path}")

if __name__ == "__main__":
    # 수정된 run_inference 함수는 폴더 경로와 이미지 개수를 명시적으로 받습니다.
    run_inference(
        folder_path=INPUT_IMAGE_FOLDER_PATH, 
        n_images=N_IMAGES_TO_INFER,
        output_base_dir=OUTPUT_BASE_DIR
    )
    