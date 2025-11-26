import cv2
import time
import os
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

THING_CLASSES = [
  "vehicle", "bus", "truck", "otherCar",
  "motorcycle", "bicycle", "pedestrian", "rider",
  "TrafficSign", "TrafficLight", "constructionGuide", "trafficDrum", 
]
STUFF_CLASSES = [
  "Freespace", "curb", "sidewalk", "crosswalk", 
  "roadMark", "whiteLane", "yellowLane", #"background"
]
# 학습 때 이 순서대로 합쳤으므로 추론 때도 순서가 같아야 합니다.
ALL_CLASSES = THING_CLASSES + STUFF_CLASSES

# 1. 설정 및 모델 로드
def setup_predictor(config_path, weights_path, score_thresh=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(config_path)  # 학습 때 사용한 config 파일 (.yaml)
    cfg.MODEL.WEIGHTS = weights_path  # 학습된 모델 가중치 (.pth)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh  # 임계값 설정
    cfg.MODEL.DEVICE = "cuda"  # GPU 사용 (없으면 "cpu")
    return DefaultPredictor(cfg), cfg

# 2. 메인 실행 함수
def run_video_inference(input_video_path, output_video_path, config_path, weights_path):
    # 모델 준비
    predictor, cfg = setup_predictor(config_path, weights_path)
    
    # [중요] Config에 기록된 데이터셋 이름을 가져옵니다 (보통 'train_dataset_name' 등)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0] if len(cfg.DATASETS.TRAIN) else "user_custom")
    
    # 여기에 학습 때 쓴 클래스 리스트를 다시 주입
    metadata.set(thing_classes=ALL_CLASSES)

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: 동영상을 열 수 없습니다.")
        return

    # 비디오 정보 가져오기
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 비디오 저장 객체 생성 (코덱: mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, input_fps, (width, height))

    print(f"Start inference on: {input_video_path}")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
        # --- 추론 시작 ---
        outputs = predictor(frame)
        
        # --- 시각화 ---
        v = Visualizer(frame[:, :, ::-1], metadata=metadata, scale=1.0, instance_mode=ColorMode.IMAGE)
        
        # 추론 결과를 이미지 위에 그리기
        out_viz = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        
        # 결과 이미지를 다시 BGR로 변환
        res_frame = out_viz.get_image()[:, :, ::-1]
        
        # --- FPS 계산 및 표시 ---
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        
        # 영상 좌측 상단에 FPS 텍스트 추가
        cv2.putText(res_frame, f"FPS: {fps:.2f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 결과 프레임 저장
        out.write(res_frame)
        
        # (선택사항) 실시간으로 화면에 띄우기 - 'q' 누르면 종료
        # cv2.imshow('Detectron2 Inference', res_frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    # 자원 해제
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done! Output saved to: {output_video_path}")

# --- 실행 ---
# 경로를 본인 환경에 맞게 수정해주세요
INPUT_VIDEO = "/home/elicer/dev/gt/data/test_data/input_video.mp4"
OUTPUT_VIDEO = "/home/elicer/dev/gt/output/inference_results/output_result.mp4"
CONFIG_FILE = "/home/elicer/dev/gt/custom_model/20251119_073846/config.yaml"    # 학습 시 생성된 config.yaml 경로
WEIGHTS_FILE = "home/elicer/dev/gt/custom_model/20251119_073846/model_final.pth" # 학습된 pth 파일 경로

# 실행
run_video_inference(INPUT_VIDEO, OUTPUT_VIDEO, CONFIG_FILE, WEIGHTS_FILE)