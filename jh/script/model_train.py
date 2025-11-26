# log_train_fixed.py

import os
import json
import logging
import detectron2
from detectron2.utils.logger import setup_logger
import matplotlib.pyplot as plt
import numpy as np
import cv2
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
import torch

# ============ logger 설정 (중복 방지, 파일+콘솔) ============
log_dir = "./train_logs"

# ===== 기존 로그 파일 삭제 =====
    
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "train.log")

if os.path.exists(log_file):
    os.remove(log_file)
    print(f"[INFO] 기존 로그 파일 삭제: {log_file}")
    
# Detectron2 기본 로거도 로그 디렉터리에 기록하게 함 (선택)
setup_logger(output=log_dir, name="detectron2")

logger = logging.getLogger("detectron2_custom")
logger.setLevel(logging.INFO)

# 이미 핸들러가 있으면 다시 추가하지 않음 (스크립트 재실행 대비)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # 콘솔
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Detectron2 기본 로거와 중복 전파 방지
logger.propagate = False

# ============ 데이터셋 등록 ============
register_coco_instances("train_parking", {}, "/home/elicer/dev/jh/data/train_subset.json", "/home/elicer/dev/jh/data/train_subset/images")
register_coco_instances("val_parking", {}, "/home/elicer/dev/jh/data/val_subset.json", "/home/elicer/dev/jh/data/val_subset/images")

val_metadata = MetadataCatalog.get("val_parking")
dataset_dicts = DatasetCatalog.get("val_parking")

# ============ 학습 설정 ============
with open('/home/elicer/dev/jh/data/train_subset.json', 'r') as f:
    data = json.load(f)
    cat_len = len(data.get('categories'))

cfg = get_cfg()
cfg.TRAIN_NAME = "train_test"
cfg.OUTPUT_DIR = os.path.join('./output', cfg.TRAIN_NAME)
cfg.merge_from_file("./detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.DATASETS.TRAIN = ("train_parking",)
cfg.DATASETS.TEST = ("val_parking",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")

# NOTE: IMS_PER_BATCH을 8로 두려면, 실제 GPU 수 * per-gpu-batch를 맞춰야 함.
cfg.SOLVER.IMS_PER_BATCH = 8

num_gpu = torch.cuda.device_count()
bs = num_gpu * 2  # (기존 코드 유지) 필요하면 수정
cfg.SOLVER.BASE_LR = 0.02 * bs / 16
cfg.TEST.EVAL_PERIOD = 100
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = cat_len

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# ============ Custom Trainer ============
class LoggingTrainer(DefaultTrainer):
    def run_step(self):
        super().run_step()
        storage = self.storage

        # 100 iteration마다 기록 (iter이 0일 때에도 기록하려면 keep 0)
        if (self.iter % 100) == 0:
            latest = storage.latest()
            # 안전하게 float 변환: tensor/np/scalar 모두 처리
            losses = {}
            for k, v in latest.items():
                try:
                    losses[k] = float(v)
                except Exception:
                    try:
                        # tensor-like
                        losses[k] = float(v.item())
                    except Exception:
                        # 변환 불가하면 건너뜀
                        continue

            if losses:
                logger.info(f"Iter {self.iter}: {losses}")
            else:
                # 보조: total_loss 키가 있을 경우 문자열로 찍기 (혹시 타입 문제 있을 때)
                if "total_loss" in latest:
                    logger.info(f"Iter {self.iter}: total_loss={latest.get('total_loss')}")

    def train(self):
        try:
            return super().train()
        finally:
            # 학습이 끝나거나 중단될 때 마지막 값 기록
            latest = self.storage.latest()
            final = {}
            for k, v in latest.items():
                try:
                    final[k] = float(v)
                except Exception:
                    try:
                        final[k] = float(v.item())
                    except Exception:
                        continue
            if final:
                logger.info(f"Final Iter {self.iter}: {final}")
            else:
                logger.info(f"Final Iter {self.iter}: (no scalars in storage.latest())")

# ============ Trainer 실행 ============
trainer = LoggingTrainer(cfg)
trainer.resume_or_load(resume=False)

logger.info("Training start!")
trainer.train()
logger.info("Training finished!")