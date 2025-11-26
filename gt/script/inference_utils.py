import cv2
import numpy as np
import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode

# --- [고정] 클래스 정의 (학습 코드와 100% 일치해야 함) ---
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

class MyPredictor:
    def __init__(self, weight_path, score_thresh=0.5):
        """
        모델을 초기화하고 메모리에 로드합니다.
        Args:
            weight_path (str): 학습된 .pth 파일 경로
            score_thresh (float): 이 점수보다 낮은 예측은 버림
        """
        self.cfg = get_cfg()
        # 모델 구조 설정 (학습때 쓴 것과 동일한 YAML)
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        
        # 가중치 로드
        self.cfg.MODEL.WEIGHTS = weight_path
        # 디바이스 설정 (GPU 있으면 'cuda', 없으면 'cpu')
        self.cfg.MODEL.DEVICE = "cuda" 
        
        # 테스트 설정
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ALL_CLASSES)
        
        # 예측 엔진 생성
        self.predictor = DefaultPredictor(self.cfg)
        
        # 시각화를 위한 메타데이터 수동 생성 (데이터셋 등록 과정 생략 위해)
        self.metadata = MetadataCatalog.get("__nonexistent_dataset_name__")
        self.metadata.thing_classes = ALL_CLASSES
        self.metadata.stuff_classes = [] # Stuff는 보통 시각화에서 제외하거나 별도 처리

    def run(self, image_bgr):
        """
        이미지 한 장을 받아 추론 결과를 시각화해서 돌려줍니다.
        Args:
            image_bgr (numpy array): OpenCV로 읽은 이미지 (BGR)
        Returns:
            vis_image (numpy array): 추론 결과가 그려진 이미지
            raw_outputs (dict): (필요시 사용) 좌표 등 원본 데이터
        """
        # 1. 추론 수행
        outputs = self.predictor(image_bgr)
        
        # 2. 시각화 (Visualizer)
        # instance_mode=ColorMode.IMAGE_BW: 배경을 흑백으로 처리해 객체 강조 (싫으면 제거)
        v = Visualizer(image_bgr[:, :, ::-1], 
                       metadata=self.metadata, 
                       scale=1.0, 
                       instance_mode=ColorMode.IMAGE_BW)
        
        # CPU로 결과 이동 후 그리기
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # 3. 결과 반환 (BGR 형태로 변환)
        return out.get_image()[:, :, ::-1]