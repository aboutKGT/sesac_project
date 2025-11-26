import os
import torch
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances

# --- 1. 설정 변수 (사용자가 제공한 경로) ---
TRAINING_OUTPUT_DIR = "/home/elicer/dev/gt/custom_model/20251119_090609"
VAL_JSON_PATH = '/home/elicer/dev/gt/data/label_data/val_coco_format.json'
VAL_IMAGE_DIR = '/home/elicer/dev/detectron2/final_data/val/images'

def setup_cfg():
    """
    학습된 설정 파일(config.yaml)을 로드합니다.
    """
    cfg = get_cfg()
    
    # 1. 학습 때 저장한 config.yaml 불러오기
    config_path = os.path.join(TRAINING_OUTPUT_DIR, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다: {config_path}")
    
    cfg.merge_from_file(config_path)
    
    # 2. 학습된 가중치(model_final.pth) 연결
    weights_path = os.path.join(TRAINING_OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = weights_path
    
    # 3. 데이터셋 설정
    cfg.DATASETS.TEST = ("my_dataset_val", )
    
    # GPU 사용 여부
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"
        
    return cfg

def main():
    logger = setup_logger()
    logger.info("모델 성능 평가를 시작합니다...")

    # 1. 검증 데이터셋 등록
    val_dataset_name = "my_dataset_val"
    try:
        register_coco_instances(val_dataset_name, {}, VAL_JSON_PATH, VAL_IMAGE_DIR)
    except AssertionError:
        pass # 이미 등록된 경우 무시

    # 2. 설정 로드
    cfg = setup_cfg()

    # 3. 모델 빌드 및 가중치 로드
    # (DefaultPredictor 대신 평가를 위해 직접 모델을 빌드합니다)
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval() # 평가 모드로 전환

    # 4. 평가기(Evaluator) 생성
    # COCOEvaluator는 bbox(박스)와 segm(마스크) 성능을 모두 계산합니다.
    evaluator = COCOEvaluator(val_dataset_name, output_dir=TRAINING_OUTPUT_DIR)
    
    # 5. 데이터 로더 생성
    val_loader = build_detection_test_loader(cfg, val_dataset_name)

    # 6. 추론 및 성능 계산 실행
    print("\n검증 데이터셋에 대한 추론 및 평가를 진행 중입니다. 잠시만 기다려주세요...")
    results = inference_on_dataset(model, val_loader, evaluator)
    
    # 7. 결과 출력
    print("\n" + "="*40)
    print(" [평가 결과 요약] ")
    print("="*40)
    
    # bbox 결과 출력
    if "bbox" in results:
        print("\n[Box Detection Performance]")
        print(results["bbox"])
        
    # segm 결과 출력 (Instance Segmentation 모델인 경우)
    if "segm" in results:
        print("\n[Instance Segmentation Performance]")
        print(results["segm"])

    print("="*40)
    print(f"상세 결과는 {TRAINING_OUTPUT_DIR} 폴더 내의 로그 파일에서도 확인 가능합니다.")

if __name__ == "__main__":
    main()