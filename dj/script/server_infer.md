`server_infer.py` 코드 개요:
파일 내용 확인 중:


[1 tool called]


네, 이 파일은 **Detectron2 추론 코드**입니다.

주요 내용:
1. **Detectron2 모듈 사용**:
   - `detectron2.config.get_cfg`
   - `detectron2.engine.DefaultPredictor`
   - `detectron2.utils.visualizer.Visualizer`

2. **기능**:
   - 학습된 모델 가중치 로드 (`model_final.pth`)
   - 이미지 폴더에서 무작위로 N장 선택
   - 각 이미지에 대해 객체 탐지 추론 수행
   - 결과 시각화 및 저장

3. **클래스 필터링**:
   - 기본적으로 모든 클래스 탐지
   - `DESIRED_CLASSES`로 특정 클래스만 필터링 가능 (현재는 `["vehicle", "pedestrian", "Freespace"]`)

4. **설정**:
   - 학습 결과물 경로: `/home/elicer/dev/dj/custom_model/20251118_094633`
   - 입력 이미지 폴더: `/home/elicer/dev/detectron2/final_data/val/images`
   - 출력 경로: `/home/elicer/dev/dj/output/inference_results`

이 코드로 학습된 Detectron2 모델을 사용해 이미지에서 객체를 탐지할 수 있습니다.
## 코드 개요

Detectron2 모델로 이미지 객체 탐지를 수행하는 추론 스크립트입니다.

### 주요 구성 요소

#### 1. 설정 변수 (15-38줄)
```15:38:dev/dj/script/server_infer.py
# 학습 결과물이 저장된 디렉토리 경로
TRAINING_OUTPUT_DIR = "/home/elicer/dev/gt/custom_model/20251118_094633" 

# 추론할 이미지들이 있는 폴더 경로
INPUT_IMAGE_FOLDER_PATH = "/home/elicer/dev/detectron2/final_data/val/images"

# 추론 결과를 저장할 기본 경로
OUTPUT_BASE_DIR = "/home/elicer/dev/gt/output/inference_results"

# 무작위로 추론할 이미지의 개수
N_IMAGES_TO_INFER = 10

# 카테고리 정의 (학습 시 사용한 THING_CLASSES와 동일해야 함)
ALL_CLASSES = [
    "vehicle", "bus", "truck", "policeCar", "ambulance", "schoolBus", "otherCar",
    "motorcycle", "bicycle", "twoWheeler", "pedestrian", "rider",
    "Freespace", "curb", "sidewalk", "crosswalk", "safetyZone", "speedBump", 
    "roadMark", "whiteLane", "yellowLane", "blueLane", "redLane", "stoplane",
    "TrafficSign", "TrafficLight", "constructionGuide", "trafficDrum", 
    "nibberCore", "warmingTriangle", "lense", "egoVehicle", "background"
]
```

#### 2. `setup_predictor()` 함수 (42-65줄)
- Detectron2 모델 초기화
- 설정 파일(`config.yaml`)과 가중치(`model_final.pth`) 로드
- 추론 임계값 설정 (SCORE_THRESH_TEST: 0.5, NMS_THRESH: 0.5)
- `DefaultPredictor` 생성 및 반환

#### 3. `load_random_images()` 함수 (67-127줄)
- 지정 폴더에서 이미지 파일 검색
- 무작위로 N장 선택
- OpenCV로 이미지 로드 (한글 경로 처리 포함)
- 로드된 이미지 리스트 반환

#### 4. `run_inference()` 함수 (129-214줄)
- 메인 추론 로직
- 단계:
  1. 모델 로드: `setup_predictor()` 호출
  2. 이미지 로드: `load_random_images()` 호출
  3. 출력 디렉토리 생성 (타임스탬프 포함)
  4. 각 이미지에 대해:
     - `predictor()`로 추론
     - 클래스 필터링 (선택적, `DESIRED_CLASSES` 사용)
     - `Visualizer`로 시각화
     - 결과 이미지 저장

#### 5. 메인 실행부 (216-222줄)
```216:222:dev/dj/script/server_infer.py
if __name__ == "__main__":
    # 수정된 run_inference 함수는 폴더 경로와 이미지 개수를 명시적으로 받습니다.
    run_inference(
        folder_path=INPUT_IMAGE_FOLDER_PATH, 
        n_images=N_IMAGES_TO_INFER,
        output_base_dir=OUTPUT_BASE_DIR
    )
```

### 실행 흐름

1. 설정된 폴더에서 이미지 10장 무작위 선택
2. 학습된 모델 로드
3. 각 이미지에 대해 객체 탐지 수행
4. 결과 시각화 (바운딩 박스, 클래스 라벨 등)
5. 타임스탬프가 포함된 폴더에 결과 저장

### 특징

- 배치 추론: 여러 이미지를 한 번에 처리
- 클래스 필터링: 특정 클래스만 표시 가능
- 한글 경로 지원: `cv2.imdecode` 사용
- 결과 자동 저장: 타임스탬프로 중복 방지

이 스크립트는 학습된 Detectron2 모델로 이미지에서 차량, 보행자, 도로 표시 등을 탐지합니다.