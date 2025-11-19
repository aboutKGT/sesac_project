Detectron2 추론 실행을 위한 설정 요약:

요약
### 1. 필수 설정 (Config 객체 생성 및 설정)

Detectron2 추론 실행을 위한 설정 요약:

## Detectron2 추론 설정 단계

### 1. 필수 설정 (Config 객체 생성 및 설정)

```python
# 1단계: 기본 설정 객체 생성
cfg = get_cfg()

# 2단계: 모델 설정 파일 로드
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# 3단계: 모델 가중치(Weights) 설정
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

# 4단계: 추론 임계값 설정
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # 신뢰도 임계값 모델이 예측한 신뢰도(confidence score)가 0.5 이상인 탐지만 유지
cfg.MODEL.ROI_HEADS.NMS_THRESH = 0.5  # NMS 임계값 겹치는 박스 중 IoU가 0.5 이상이면 하나만 남기고 나머지 제거
```
네, 맞습니다. 다만 두 임계값의 의미가 다릅니다:


[1 tool called]


## 두 임계값의 의미

### 1. `SCORE_THRESH_TEST = 0.5` (신뢰도 임계값)

```python
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
```

- 의미: 모델이 예측한 신뢰도(confidence score)가 0.5 이상인 탐지만 유지
- 동작:
  - 신뢰도 ≥ 0.5 → 유지
  - 신뢰도 < 0.5 → 제거
- 예시:
  - person: 0.8 → 유지
  - car: 0.3 → 제거
  - bicycle: 0.6 → 유지

### 2. `NMS_THRESH = 0.5` (NMS IoU 임계값)

```python
cfg.MODEL.ROI_HEADS.NMS_THRESH = 0.5
```

- 의미: 겹치는 박스 중 IoU(Intersection over Union)가 0.5 이상이면 하나만 남기고 나머지 제거
- 동작:
  - 같은 객체에 대해 여러 박스가 겹칠 때
  - IoU ≥ 0.5 → 신뢰도가 가장 높은 박스만 유지, 나머지 제거
  - IoU < 0.5 → 다른 객체로 간주하여 모두 유지
- 예시:
  - 같은 person에 대해 3개 박스가 겹침 (IoU 0.7, 0.6)
  - 신뢰도가 가장 높은 박스만 유지, 나머지 2개 제거

## 요약

| 임계값 | 의미 | 0.5 이하 처리 |
|--------|------|---------------|
| `SCORE_THRESH_TEST = 0.5` | 신뢰도 임계값 | 제거 (0.5 미만 탐지 결과 버림) |
| `NMS_THRESH = 0.5` | IoU 임계값 | 유지 (다른 객체로 간주) |

## 값 조정 시 영향

### `SCORE_THRESH_TEST` 조정
- 0.5 → 0.3: 더 많은 탐지 (낮은 신뢰도도 포함, 오탐 증가 가능)
- 0.5 → 0.7: 더 적은 탐지 (높은 신뢰도만, 정확도 향상)

### `NMS_THRESH` 조정
- 0.5 → 0.3: 더 많은 중복 박스 제거 (엄격한 NMS)
- 0.5 → 0.7: 더 적은 중복 박스 제거 (느슨한 NMS)

**결론**: `SCORE_THRESH_TEST = 0.5`는 신뢰도 0.5 미만 탐지를 제거합니다. `NMS_THRESH = 0.5`는 겹침이 큰 박스들을 정리하는 IoU 임계값입니다.
### 2. 메타데이터 설정

```python
# 5단계: 클래스 정보 메타데이터 로드
metadata = MetadataCatalog.get("coco_2017_val")  # COCO의 경우
# 또는 커스텀 모델의 경우:
# MetadataCatalog.get("custom_inference").thing_classes = THING_CLASSES
```
메타데이터(Metadata)의 역할과 필요성 설명:


[2 tools called]


## 메타데이터(Metadata)란?

메타데이터는 모델이 탐지한 객체의 클래스 정보를 담은 데이터입니다. 모델은 숫자 ID만 출력하므로, 이를 사람이 읽을 수 있는 클래스명으로 변환하는 데 필요합니다.

### 메타데이터가 필요한 이유

```python
# 모델이 출력하는 것 (숫자 ID만)
outputs["instances"].pred_classes = tensor([0, 2, 5, 0, ...])  # 0, 2, 5는 뭘까?

# 메타데이터가 제공하는 것 (클래스명 매핑)
metadata.thing_classes = ["person", "bicycle", "car", "motorcycle", ...]
# 0 → "person"
# 2 → "car"  
# 5 → "bus"
```

### 메타데이터에 포함된 정보

1. 클래스 이름 목록 (`thing_classes`)
   - 숫자 ID → 클래스명 매핑
   - 예: `["person", "bicycle", "car", "motorcycle", ...]`

2. 클래스별 색상 정보
   - 시각화 시 각 클래스마다 다른 색상 사용

3. 기타 정보
   - 클래스 설명, 카테고리 등

### 코드에서의 사용 예시

```python
# 1. 메타데이터 로드
metadata = MetadataCatalog.get("coco_2017_val")
# COCO의 80개 클래스 정보가 자동으로 로드됨

# 2. 시각화에 사용
v = Visualizer(im, metadata=metadata)  # 메타데이터를 전달
out = v.draw_instance_predictions(outputs["instances"])
# → 바운딩 박스에 "person", "car" 같은 클래스명이 표시됨
```

### COCO vs 커스텀 모델

#### COCO 모델 (현재 코드)
```python
# COCO는 이미 등록되어 있어서 간단
metadata = MetadataCatalog.get("coco_2017_val")
# 80개 클래스가 자동으로 로드됨
```

#### 커스텀 모델 (server_infer.py)
```python
# 커스텀 클래스를 직접 등록해야 함
THING_CLASSES = ["vehicle", "pedestrian", "Freespace", ...]

metadata_name = "custom_inference"
MetadataCatalog.get(metadata_name).thing_classes = THING_CLASSES
metadata = MetadataCatalog.get(metadata_name)
```

### 메타데이터 없이 추론하면?

```python
# 메타데이터 없이 추론 가능
outputs = predictor(image)  # ✅ 추론은 됨

# 하지만 시각화 불가능
v = Visualizer(im)  # ❌ 클래스명을 모르므로 라벨 표시 불가
```

### 요약

| 항목 | 설명 |
|------|------|
| **메타데이터** | 클래스 ID를 클래스명으로 변환하는 정보 |
| **포함 내용** | 클래스 이름 목록, 색상 정보 등 |
| **주요 용도** | 결과 시각화 (바운딩 박스에 클래스명 표시) |
| **COCO** | 이미 등록되어 있어서 `get()`만 호출 |
| **커스텀** | 직접 클래스 목록을 등록해야 함 |

**결론**: 메타데이터는 모델이 출력한 숫자 ID(0, 1, 2...)를 사람이 읽을 수 있는 클래스명("person", "car"...)으로 변환하는 정보입니다. 시각화할 때 필수입니다.

### 3. Predictor 초기화

```python
# 6단계: 추론기 생성
predictor = DefaultPredictor(cfg)
```

### 4. 추론 실행

```python
# 7단계: 이미지 추론
outputs = predictor(image)

# 8단계: 결과 처리
instances = outputs["instances"]  # 탐지된 객체들
```

## YOLO vs Detectron2 비교

| 항목 | YOLO | Detectron2 |
|------|------|------------|
| 모델 로드 | `YOLO('yolov8n.pt')` 한 줄 | Config + Weights + Metadata 여러 단계 |
| 설정 파일 | 없음 (모델에 포함) | YAML 설정 파일 필요 |
| 클래스 정보 | 모델에 내장 | 별도 MetadataCatalog 설정 필요 |
| 추론 코드 | `model(image)` | `predictor(image)` + 결과 처리 |
| 임계값 설정 | 간단한 파라미터 | Config 객체의 여러 속성 설정 |

## Detectron2가 복잡한 이유

### 1. Config 시스템
- 모든 설정이 Config 객체에 저장됨
- YAML 파일로 관리
- 모델 구조, 학습 파라미터, 추론 설정 등 포함

### 2. 메타데이터 분리
- 클래스 정보가 모델과 분리
- `MetadataCatalog`로 별도 관리
- 시각화에 필요

### 3. 모듈화된 구조
- Config, Weights, Metadata가 분리
- 유연하지만 설정 단계가 많음

### 4. 다양한 모델 지원
- Faster R-CNN, RetinaNet, Mask R-CNN 등
- 각 모델마다 다른 설정 필요

## 간단한 사용 예시

```python
# COCO 모델 사용 (가장 간단)
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)
outputs = predictor(image)
```

## 커스텀 모델 사용 시 추가 설정

```python
# 커스텀 모델의 경우 더 복잡
cfg.merge_from_file("path/to/config.yaml")  # 커스텀 설정 파일
cfg.MODEL.WEIGHTS = "path/to/model_final.pth"  # 커스텀 가중치

# 클래스 정보 직접 등록
from detectron2.data import MetadataCatalog
MetadataCatalog.get("custom").thing_classes = ["class1", "class2", ...]
metadata = MetadataCatalog.get("custom")
```

## 요약

Detectron2는 다음 3가지를 설정해야 합니다:
1. Config: 모델 구조 및 설정
2. Weights: 학습된 가중치
3. Metadata: 클래스 정보

YOLO는 이 3가지가 모델 파일에 통합되어 있어 한 줄로 로드할 수 있지만, Detectron2는 각각 설정해야 합니다. 대신 더 세밀한 제어가 가능합니다.

현재 코드(`cocos_infer.py`)는 COCO 모델을 사용하므로 `setup_predictor()` 함수에서 대부분 자동 처리됩니다.
## `cocos_infer.py` 코드 개요

Detectron2의 COCO 사전 학습 모델로 이미지 객체 탐지를 수행하는 추론 스크립트입니다.

### 주요 구성 요소

#### 1. 설정 변수 (16-32줄)
```22:31:dev/dj/script/cocos_infer.py
COCO_MODEL_CONFIG = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"  # 사용할 COCO 모델 설정 파일명

# 추론할 이미지들이 있는 폴더 경로
INPUT_IMAGE_FOLDER_PATH = "/home/elicer/dev/detectron2/final_data/val/images"  # 입력 이미지 폴더 경로

# 추론 결과를 저장할 기본 경로 (하위에 이미지별 폴더가 생성됩니다)
OUTPUT_BASE_DIR = "/home/elicer/dev/dj/output/inference_results"  # 결과 저장 기본 디렉토리

# 무작위로 추론할 이미지의 개수 (N)
N_IMAGES_TO_INFER = 10  # 원하는 이미지 개수로 설정하세요.
```
- COCO 모델 설정, 입력 이미지 경로, 출력 경로, 추론할 이미지 개수

#### 2. `setup_predictor()` 함수 (35-56줄)
- COCO 사전 학습 모델 초기화
- `model_zoo`에서 설정과 가중치 자동 다운로드
- COCO 메타데이터 로드 (80개 클래스)
- 추론 임계값 설정 (신뢰도 0.5, NMS 0.5)

#### 3. `load_random_images()` 함수 (58-118줄)
- 지정 폴더에서 이미지 파일 검색
- 무작위로 N장 선택
- OpenCV로 이미지 로드 (한글 경로 처리 포함)
- 로드된 이미지 리스트 반환

#### 4. `run_inference()` 함수 (120-197줄)
- 메인 추론 로직
- 단계:
  1. COCO 모델 로드
  2. 이미지 로드 (무작위 N장)
  3. 출력 디렉토리 생성 (타임스탬프 포함)
  4. 각 이미지에 대해:
     - 객체 탐지 추론
     - 클래스 필터링 (선택적, `DESIRED_CLASSES` 사용)
     - 결과 시각화 (바운딩 박스, 라벨)
     - 결과 이미지 저장

#### 5. 메인 실행부 (199-205줄)
```199:205:dev/dj/script/cocos_infer.py
if __name__ == "__main__":  # 스크립트가 직접 실행될 때만 실행
    # 수정된 run_inference 함수는 폴더 경로와 이미지 개수를 명시적으로 받습니다.
    run_inference(
        folder_path=INPUT_IMAGE_FOLDER_PATH,  # 입력 이미지 폴더 경로
        n_images=N_IMAGES_TO_INFER,  # 추론할 이미지 개수
        output_base_dir=OUTPUT_BASE_DIR  # 결과 저장 기본 경로
    )
```

### 실행 흐름

1. COCO 모델 자동 다운로드 및 로드 (최초 1회)
2. 지정 폴더에서 이미지 10장 무작위 선택
3. 각 이미지에 대해 COCO 모델로 객체 탐지
4. 결과 시각화 (바운딩 박스, 클래스 라벨, 신뢰도)
5. 타임스탬프가 포함된 폴더에 결과 저장

### 특징

- COCO 사전 학습 모델 사용: 별도 학습 없이 바로 사용
- 80개 COCO 클래스 탐지: person, car, bicycle, bus, truck 등
- 클래스 필터링: 원하는 클래스만 표시 가능
- 배치 처리: 여러 이미지를 한 번에 처리
- 한글 경로 지원: `cv2.imdecode` 사용
- 결과 자동 저장: 타임스탬프로 중복 방지

### COCO 모델과 커스텀 모델의 차이

- `cocos_infer.py`: COCO 사전 학습 모델 사용 (80개 일반 객체 클래스)
- `server_infer.py`: 커스텀 학습 모델 사용 (도로/교통 관련 35개 클래스)

이 스크립트는 COCO 데이터셋으로 학습된 모델을 사용해 일반적인 객체 탐지를 수행합니다.