import os
import cv2
import random
import numpy as np
import torch
import datetime
import json
import logging

# Detectron2 관련 임포트
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
import detectron2.utils.comm as comm
from detectron2.engine import DefaultPredictor

from detectron2.evaluation import COCOEvaluator


# --- 1. 설정 ---
# 100% 전체 데이터 JSON 경로
TRAIN_JSON_PATH = '/home/elicer/dev/gt/data/label_data/train_coco_full.json' 
TRAIN_IMAGE_ROOT = '/home/elicer/dev/detectron2/final_data/train/images'

thistime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = '/home/elicer/dev/gt/custom_model/' + thistime

# 클래스 정의
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

# --- 2. Custom Hook: 검증(Validation) Loss 계산용 ---
class ValidationLossHook(HookBase):
    """
    학습 중간에 Validation Dataset으로 Loss를 계산하여 로그에 남기는 훅
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        
        self._loader = iter(build_detection_train_loader(
            self.cfg, 
            num_workers=0
        ))
        
    def after_step(self):
        if (self.trainer.iter + 1) % self.cfg.TEST.EVAL_PERIOD != 0:
            return
        
        try:
            data = next(self._loader)
        except StopIteration:
            self._loader = iter(build_detection_train_loader(
                self.cfg, 
                num_workers=0
            ))
            data = next(self._loader)
            
        with torch.no_grad():
            self.trainer.model.train() # Loss 계산 모드
            loss_dict = self.trainer.model(data)
            
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            total_loss = sum(loss for loss in loss_dict_reduced.values())
            loss_dict_reduced["val_total_loss"] = total_loss

            if comm.is_main_process():
                self.trainer.storage.put_scalars(**loss_dict_reduced)

# --- 3. Custom Trainer ---
class MyTrainer(DefaultTrainer):
    """
    기본 Trainer를 상속받아 Custom Hook 추가
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        # 보고서용: 저장 경로에 'inference' 폴더를 만들어 평가 결과를 저장
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        
        # COCOEvaluator: COCO 방식으로 mAP(정확도)를 채점해주는 도구
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    def build_hooks(self):
        # 1. 부모 클래스(DefaultTrainer)의 기본 훅들 호출
        hooks = super().build_hooks()
        
        # 2. ValidationLossHook을 리스트 뒤에서 두 번째에 삽입
        hooks.insert(-1, ValidationLossHook(self.cfg))
        
        return hooks

# --- 4. 데이터 등록 및 분할 ---
def register_split_datasets():
    """
    전체 데이터 중 고정된 개수(예: 1000장)만 Val로 쓰고, 
    나머지는 모두 Train에 몰아주는 방식
    """
    full_name = "my_dataset_full_source"
    register_coco_instances(full_name, {}, TRAIN_JSON_PATH, TRAIN_IMAGE_ROOT)
    
    full_dicts = DatasetCatalog.get(full_name)
    
    # 랜덤 셔플
    random.seed(42)
    random.shuffle(full_dicts)
    
    # val 데이터 개수 고정
    VAL_SET_SIZE = 50 
    
    # 혹시 데이터가 200장보다 적을 경우를 대비한 안전장치
    if len(full_dicts) < VAL_SET_SIZE:
        VAL_SET_SIZE = int(len(full_dicts) * 0.1) # 데이터가 너무 적으면 10%만

    # 앞부분 1000개를 검증용으로, 나머지를 학습용으로
    val_dicts = full_dicts[:VAL_SET_SIZE]
    train_dicts = full_dicts[VAL_SET_SIZE:]
    
    # Train 등록
    DatasetCatalog.register("my_dataset_train", lambda: train_dicts)
    MetadataCatalog.get("my_dataset_train").set(thing_classes=ALL_CLASSES, image_root=TRAIN_IMAGE_ROOT, evaluator_type="coco")
    
    # Val 등록
    DatasetCatalog.register("my_dataset_val", lambda: val_dicts)
    MetadataCatalog.get("my_dataset_val").set(thing_classes=ALL_CLASSES, image_root=TRAIN_IMAGE_ROOT, evaluator_type="coco")
    
    return "my_dataset_train", "my_dataset_val"

# --- 5. Config 설정 ---
def setup_config(train_name, val_name):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # 데이터셋 지정
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)  # 검증용 데이터셋 등록
    cfg.DATALOADER.NUM_WORKERS = 16
    
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    
    # 학습 파라미터
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.BASE_LR = 0.002
    cfg.SOLVER.MAX_ITER = 2000
    cfg.SOLVER.STEPS = (1500, )
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000 # step마다 모델 저장
    
    # 검증 주기 설정 (이 주기마다 val_loss 기록)
    cfg.TEST.EVAL_PERIOD = 1000
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ALL_CLASSES)
    
    # 클래스 불균형 해소 (RepeatFactorTrainingSampler)
    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.001 
    
    cfg.OUTPUT_DIR = OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main():
    # logger에 저장 경로(OUTPUT_DIR) 전달
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logger(output=OUTPUT_DIR) 
    
    # 1. 데이터셋 등록 (Train/Val 분리)
    train_name, val_name = register_split_datasets()
    
    # 2. 설정 로드
    cfg = setup_config(train_name, val_name)
    
    # 설정 파일 저장
    with open(os.path.join(cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
        f.write(cfg.dump())
        
    # 3. 학습 시작 (Custom Trainer 사용)
    # 이제부터 출력되는 모든 로그가 log.txt에도 기록됩니다.
    print(f"학습 시작! 로그는 {cfg.OUTPUT_DIR}/log.txt 에 저장됩니다.")
    
    trainer = MyTrainer(cfg) 
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    # [보고서용] 학습 완료 후 시각화 결과 저장
    print("\n[보고서용] 학습 완료 후 시각화 결과 저장 중...")
    
    # 방금 학습한 가중치 로드
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    
    # [중요] 테스트용이라 100번만 돌렸으므로, 확신이 낮아도 일단 그리도록 설정
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  
    
    # 예측기 생성
    predictor = DefaultPredictor(cfg)
    
    # 검증 데이터셋에서 랜덤 5장 뽑아서 그림 그리기
    val_loader = DatasetCatalog.get("my_dataset_val")
    
    # 혹시 데이터가 5장보다 적을 경우를 대비해 min 사용
    sample_count = min(5, len(val_loader))
    
    for d in random.sample(val_loader, sample_count):    
        im = cv2.imread(d["file_name"])
        if im is None: continue # 이미지 경로 에러 방지
        
        outputs = predictor(im)
        
        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get("my_dataset_val"), 
                       scale=0.8, 
                       instance_mode=ColorMode.IMAGE_BW
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # 결과 저장
        save_path = os.path.join(cfg.OUTPUT_DIR, f"result_{d['image_id']}.jpg")
        cv2.imwrite(save_path, out.get_image()[:, :, ::-1])
        print(f" - 저장됨: {save_path}")
    

if __name__ == "__main__":
    main()