import os
import cv2
import random
import numpy as np
import torch
import datetime
import json

from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances, load_coco_json
from detectron2.utils.visualizer import Visualizer

# --- 1. 설정 및 데이터셋 등록 ---

# [사용자 설정]
TRAIN_JSON_PATH = '/home/elicer/dev/gt/data/label_data/train_coco_format.json'
TRAIN_IMAGE_ROOT = '/home/elicer/dev/detectron2/final_data/train/images'
thistime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = '/home/elicer/dev/gt/custom_model/' + thistime

# [중요] 사용할 데이터 비율 설정 (0.0 ~ 1.0)
# 예: 0.1 = 10%, 0.5 = 50%, 1.0 = 100%(전체)
TRAIN_DATA_PERCENTAGE = 0.7 

# 클래스 정의
THING_CLASSES = [
  "vehicle", "bus", "truck", "otherCar",
  "motorcycle", "bicycle", "pedestrian", "rider",
  "TrafficSign", "TrafficLight", "constructionGuide", "trafficDrum", 
]
STUFF_CLASSES = [
  "Freespace", "curb", "sidewalk", "crosswalk", 
  "roadMark", "whiteLane", "yellowLane", #"background"
]
ALL_CLASSES = THING_CLASSES + STUFF_CLASSES

def register_custom_datasets():
  """
  전체 데이터를 등록한 뒤, 설정된 비율만큼 샘플링하여 학습용 데이터셋으로 재등록
  """
  # 1. 원본 전체 데이터셋 등록 (이름: my_dataset_full)
  full_dataset_name = "my_dataset_full"
  register_coco_instances(full_dataset_name, {}, TRAIN_JSON_PATH, TRAIN_IMAGE_ROOT)
  
  # 2. 전체 데이터 로드
  full_dicts = DatasetCatalog.get(full_dataset_name)
  
  # 3. 데이터 섞기
  # 랜덤하게 섞어야 특정 비디오나 구간의 데이터만 뽑히는 것을 방지
  random.seed(42) # 결과를 재현 가능하게 하려면 고정, 아니면 제거
  random.shuffle(full_dicts)
  
  # 4. 비율에 맞춰 자르기
  num_samples = int(len(full_dicts) * TRAIN_DATA_PERCENTAGE)
  dataset_dicts = full_dicts[:num_samples]
  
  print(f"\n[데이터셋 설정] 전체 {len(full_dicts)}장 중 {TRAIN_DATA_PERCENTAGE*100}% 인 {len(dataset_dicts)}장을 사용하여 학습합니다.\n")
  
  # 5. 실제 학습에 사용할 서브셋 등록 (이름: my_dataset_train)
  # 이미 로드된 dict 리스트를 반환하는 lambda 함수를 등록합니다.
  train_dataset_name = "my_dataset_train"
  DatasetCatalog.register(train_dataset_name, lambda: dataset_dicts)
  
  # 6. 메타데이터 복사 및 설정
  # 원본 데이터셋의 메타데이터(이미지 경로 등)는 가져오지 못할 수 있으므로 클래스 정보를 수동 설정
  metadata = MetadataCatalog.get(train_dataset_name)
  metadata.set(thing_classes=ALL_CLASSES)
  metadata.set(image_root=TRAIN_IMAGE_ROOT) # 시각화를 위해 필요
  metadata.set(evaluator_type="coco")
  
  return train_dataset_name

def setup_config(train_dataset_name):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  
  # 데이터셋 설정
  cfg.DATASETS.TRAIN = (train_dataset_name,)
  cfg.DATASETS.TEST = () 
  cfg.DATALOADER.NUM_WORKERS = 4
  
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
  
  cfg.SOLVER.IMS_PER_BATCH = 8
  cfg.SOLVER.BASE_LR = 0.001
  
  cfg.SOLVER.MAX_ITER = 60000 
  cfg.SOLVER.STEPS = (40000, 50000)
  
  cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ALL_CLASSES)
  
  cfg.SOLVER.CHECKPOINT_PERIOD = 10000
  
  cfg.OUTPUT_DIR = OUTPUT_DIR
  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
  return cfg

def main():
  setup_logger()
  
  # 데이터셋 등록 및 샘플링 수행
  target_dataset_name = register_custom_datasets()
  
  # 설정 로드
  cfg = setup_config(target_dataset_name)
  
  # [추가된 코드] 1. 설정을 YAML 파일로 명시적 저장
  config_save_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
  with open(config_save_path, "w") as f:
      f.write(cfg.dump())
  print(f"설정 파일(config.yaml)이 저장되었습니다: {config_save_path}")
  
  # --- 데이터 로드 확인 (샘플링된 데이터에서 확인) ---
  print("샘플링된 데이터 확인 중...")
  # lambda로 등록했으므로 다시 get으로 호출
  dataset_dicts = DatasetCatalog.get(target_dataset_name)
  metadata = MetadataCatalog.get(target_dataset_name)
  
  # 샘플 이미지가 데이터셋 크기보다 클 경우 예외 처리
  sample_count = min(3, len(dataset_dicts))
  for d in random.sample(dataset_dicts, sample_count):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    vis_path = os.path.join(cfg.OUTPUT_DIR, f"check_data_{d['image_id']}.jpg")
    cv2.imwrite(vis_path, out.get_image()[:, :, ::-1])
  print(f"데이터 확인 이미지 저장 완료: {cfg.OUTPUT_DIR}")

  # --- 학습 시작 ---
  print("학습을 시작합니다...")
  trainer = DefaultTrainer(cfg) 
  trainer.resume_or_load(resume=False)
  trainer.train()
  print("학습 완료!")

if __name__ == "__main__":
  main()