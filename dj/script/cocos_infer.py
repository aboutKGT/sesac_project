import os  # 파일 시스템 경로 조작
import cv2  # 이미지 읽기/쓰기 및 처리
import datetime  # 타임스탬프 생성
import numpy as np  # 배열 연산
import random  # 무작위 이미지 선택
import torch  # 텐서 기반 필터링을 위해 추가
from typing import List, Optional  # 타입 힌팅

# Detectron2 관련 모듈
from detectron2.config import get_cfg  # 설정 파일 로드
from detectron2.engine import DefaultPredictor  # 추론 엔진
from detectron2.utils.visualizer import Visualizer  # 결과 시각화
from detectron2.data import MetadataCatalog  # 메타데이터 관리
from detectron2 import model_zoo  # 사전 학습 모델 다운로드

# --- 1. 설정 변수 (사용자 환경에 맞게 수정 필수!) ---

# COCO 모델 설정 (다른 모델로 변경 가능)
# 예: "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
#     "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
#     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
COCO_MODEL_CONFIG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"  # 사용할 COCO 모델 설정 파일명

# 추론할 이미지들이 있는 폴더 경로
INPUT_IMAGE_FOLDER_PATH = "/home/elicer/dev/detectron2/final_data/val/images"  # 입력 이미지 폴더 경로

# 추론 결과를 저장할 기본 경로 (하위에 이미지별 폴더가 생성됩니다)
OUTPUT_BASE_DIR = "/home/elicer/dev/dj/output/inference_results"  # 결과 저장 기본 디렉토리

# 무작위로 추론할 이미지의 개수 (N)
N_IMAGES_TO_INFER = 10  # 원하는 이미지 개수로 설정하세요.
# -----------------------------------------------------------------


def setup_predictor(model_config_name):
    """Detectron2 COCO 사전 학습 모델 예측기(Predictor)와 메타데이터를 설정하고 반환합니다."""
    # 1. 설정 (Config) 로드 - COCO 모델 설정 사용
    cfg = get_cfg()  # 기본 설정 객체 생성
    cfg.merge_from_file(model_zoo.get_config_file(model_config_name))  # COCO 모델 설정 파일 로드
    
    # 2. COCO 사전 학습 가중치(Weights) 로드
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_config_name)  # COCO 사전 학습 가중치 URL 설정
    
    # 3. 추론 시 사용할 최소 임계값 설정
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 신뢰도 0.5 이상만 탐지
    
    # 4. NMS 임계값 설정
    cfg.MODEL.ROI_HEADS.NMS_THRESH = 0.5  # Non-Maximum Suppression 임계값
    
    # 5. COCO 메타데이터 가져오기 (COCO 클래스 정보 포함)
    metadata = MetadataCatalog.get("coco_2017_val")  # COCO 2017 validation 메타데이터 (80개 클래스)
    
    # 6. DefaultPredictor 초기화
    predictor = DefaultPredictor(cfg)  # 추론기 생성
    
    return predictor, metadata  # 추론기와 메타데이터 반환

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
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')  # 지원하는 이미지 확장자
    all_image_paths = []  # 모든 이미지 경로 저장 리스트

    # 1. 유효성 검사 및 경로 목록 생성
    if not os.path.isdir(folder_path):  # 폴더 경로 유효성 검사
        print(f"오류: 지정된 경로 '{folder_path}'는 유효한 폴더가 아닙니다.")
        return None
        
    try:
        for filename in os.listdir(folder_path):  # 폴더 내 모든 파일 순회
            if filename.lower().endswith(image_extensions):  # 이미지 확장자 확인
                full_path = os.path.join(folder_path, filename)  # 전체 경로 생성
                all_image_paths.append(full_path)  # 경로 리스트에 추가
    except Exception as e:  # 폴더 읽기 오류 처리
        print(f"오류: 폴더 내용을 읽는 중 문제가 발생했습니다. {e}")
        return None

    # 2. 무작위 샘플 선택
    if not all_image_paths:  # 이미지 파일이 없는 경우
        print("경고: 폴더에 로드할 수 있는 이미지 파일이 없습니다.")
        return []

    num_available = len(all_image_paths)  # 사용 가능한 이미지 개수
    if n_images <= 0:  # 요청 개수 유효성 검사
        print("경고: 로드할 이미지 개수(n)는 1 이상이어야 합니다.")
        return []
    
    if n_images > num_available:  # 요청 개수가 사용 가능한 개수보다 많은 경우
        print(f"경고: 요청한 {n_images}장이 전체 {num_available}장보다 많습니다. 전체 이미지를 로드합니다.")
        n_images = num_available  # 사용 가능한 개수로 조정

    # 무작위로 n_images 개수만큼 경로 선택 (중복 없음)
    random_paths = random.sample(all_image_paths, n_images)  # 무작위 샘플링

    # 3. 반복문으로 이미지 로드
    loaded_images = []  # 로드된 이미지 저장 리스트
    print(f"총 {n_images}장의 이미지를 무작위로 로드합니다...")
    
    for i, image_path in enumerate(random_paths):  # 선택된 이미지 경로 순회
        # 한글 경로 문제 방지를 위해 imdecode 사용
        # img_array = np.fromfile(image_path, np.uint8)  # 파일을 바이트 배열로 읽기  cv2.imread()는 한글경로 실패가능능
        # im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # 바이트 배열을 이미지로 디코딩
        im = cv2.imread(image_path)  # 이미지 파일 읽기

        if im is not None:  # 이미지 로드 성공
            loaded_images.append(im)  # 리스트에 추가
        else:  # 이미지 로드 실패
            print(f"[{i+1}/{n_images}] 로드 실패: {os.path.basename(image_path)}")

    print("--- 이미지 로드 완료 ---")
    return loaded_images  # 로드된 이미지 리스트 반환

def run_inference(folder_path: str, n_images: int, output_base_dir: str):
    """
    무작위로 n장의 이미지를 로드하여 모델 추론을 실행하고 결과를 저장합니다.
    
    Args:
        folder_path (str): 이미지가 있는 폴더 경로.
        n_images (int): 추론할 무작위 이미지 개수.
        output_base_dir (str): 결과 이미지를 저장할 최상위 폴더 경로.
    """
    
    # 1. 모델 및 메타데이터 설정 (COCO 사전 학습 모델)
    predictor, metadata = setup_predictor(COCO_MODEL_CONFIG)  # COCO 모델 로드 및 초기화
    print(f"COCO 모델 로드 완료: {COCO_MODEL_CONFIG}")
    
    # 2. 이미지 로드 (n_images 개수만큼 로드)
    # load_random_images는 NumPy 배열(이미지 데이터)의 리스트를 반환합니다.
    loaded_images = load_random_images(folder_path, n_images)  # 무작위 이미지 로드
    
    if not loaded_images:  # 이미지 로드 실패 시
        print(f"오류: 로드된 이미지가 없으므로 추론을 실행할 수 없습니다. 경로를 확인해주세요: {folder_path}")
        return

    # 결과 저장 디렉토리 생성 (타임스탬프를 추가하여 덮어쓰기 방지)
    thistime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 현재 시간을 문자열로 변환
    output_dir = os.path.join(output_base_dir, f"inference_batch_{thistime}")  # 출력 디렉토리 경로 생성
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리 생성 (이미 있으면 무시)
    print(f"결과 이미지는 다음 폴더에 저장됩니다: {output_dir}")

    # 3. 로드된 각 이미지에 대해 추론 실행
    for i, im in enumerate(loaded_images):  # 각 이미지 순회
        
        # 3-1. 추론 실행
        outputs = predictor(im)  # 모델로 객체 탐지 수행
        
        # 3-2. 클래스 필터링 (원하는 COCO 클래스만 시각화)
        # COCO 클래스 예시: "person", "bicycle", "car", "motorcycle", "bus", "truck" 등
        # 필터링을 원하지 않으면 이 부분을 주석 처리하세요
        DESIRED_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]  # 표시할 클래스 목록
        
        instances = outputs["instances"].to("cpu")  # GPU 텐서를 CPU로 이동
        
        if DESIRED_CLASSES and metadata.thing_classes:  # 필터링할 클래스가 있고 메타데이터가 있는 경우
            try:
                # 원하는 클래스 ID 리스트를 PyTorch 텐서로 생성
                desired_ids = torch.tensor(
                    [metadata.thing_classes.index(cls_name) for cls_name in DESIRED_CLASSES if cls_name in metadata.thing_classes],  # 클래스명을 ID로 변환
                    dtype=torch.long  # 정수형 텐서
                )
                
                if len(desired_ids) > 0:  # 유효한 클래스 ID가 있는 경우
                    # 예측된 클래스 ID를 가져옵니다.
                    pred_classes = instances.pred_classes  # 예측된 클래스 ID 텐서
                    
                    # torch.isin을 사용하여 필터 마스크를 생성하고 인스턴스를 필터링합니다.
                    filter_mask = torch.isin(pred_classes, desired_ids)  # 원하는 클래스만 True인 마스크 생성
                    instances = instances[filter_mask]  # 필터링된 인스턴스만 선택
            except (ValueError, AttributeError) as e:  # 필터링 오류 처리
                print(f"클래스 필터링 중 오류 발생 (모든 클래스 표시): {e}")
        
        # 필터링된 인스턴스를 다시 outputs에 할당
        outputs["instances"] = instances  # 필터링된 결과로 업데이트
        
        # 4. 결과 시각화
        # Visualizer는 RGB 입력을 기대하므로 BGR인 im을 변환하여 전달
        v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=1.0)  # BGR→RGB 변환 후 시각화 객체 생성
        
        out = v.draw_instance_predictions(outputs["instances"])  # 바운딩 박스와 라벨 그리기
        
        # 5. 결과 저장 (각 이미지마다 고유한 파일명 사용)
        # RGB 결과를 다시 BGR로 변환하여 OpenCV로 저장
        result_img = out.get_image()[:, :, ::-1]  # RGB→BGR 변환 (OpenCV 형식)
        
        output_filename = f"result_image_{i+1:02d}.jpg"  # 파일명 생성 (01, 02, ... 형식)
        output_path = os.path.join(output_dir, output_filename)  # 전체 저장 경로 생성
        
        cv2.imwrite(output_path, result_img)  # 이미지 파일로 저장
        
        print(f"✅ 이미지 {i+1}/{n_images} 추론 완료. 저장 위치: {output_path}")

if __name__ == "__main__":  # 스크립트가 직접 실행될 때만 실행
    # 수정된 run_inference 함수는 폴더 경로와 이미지 개수를 명시적으로 받습니다.
    run_inference(
        folder_path=INPUT_IMAGE_FOLDER_PATH,  # 입력 이미지 폴더 경로
        n_images=N_IMAGES_TO_INFER,  # 추론할 이미지 개수
        output_base_dir=OUTPUT_BASE_DIR  # 결과 저장 기본 경로
    )