## 📁 1. 데이터셋 및 전처리 (Data Preprocessing)

본 프로젝트는 모델 학습 및 평가를 위해 데이터셋을 표준 포맷인 **COCO 형식**으로 변환하고 특정 조건을 기준으로 검증(Validation) 세트를 구성합니다.

### 1.1. 데이터셋 구조

* **원본 데이터셋:** 원본 데이터는 특정 환경(예: Cityscape, ORIGIN)에서 수집되었으며, 학습 전 COCO 포맷으로 변환이 필요합니다.

### 1.2. 전처리 스크립트 목록

다음 스크립트들은 원본 데이터셋을 처리하고 학습/검증 세트를 구성하는 데 사용됩니다.

| 스크립트 파일명 | 주요 기능 | 목적 |
| :--- | :--- | :--- |
| **`pre_city_to_coco.py`** | **Cityscape 데이터셋 변환** | Cityscape와 같은 원본 데이터 포맷을 표준 **COCO 포맷**으로 변환합니다. |
| **`pre_origin_to_coco.py`** | **ORIGIN 데이터셋 변환** | 자체 제작/수집한 데이터셋(`origin`)을 표준 **COCO 포맷**으로 변환합니다. |
| **`pre_city_plus_dist1.py`** | Cityscape 데이터에 **거리(Depth) 정보 1**을 추가하여 전처리합니다. | 3차원 객체 인식 또는 Depth 정보를 활용한 모델 학습을 위한 데이터셋 구성. |
| **`pre_city_plus_dist2.py`** | Cityscape 데이터에 **거리(Depth) 정보 2**를 추가하여 전처리합니다. | |
| **`pre_origin_plus_dist.py`** | ORIGIN 데이터에 **거리(Depth) 정보**를 추가하여 전처리합니다. | |
| **`create_val_clear.py`** | **클리어(Clear) 조건의 검증 데이터셋 생성** | 정상적인 주행 환경(안개, 비 없음) 조건의 이미지를 검증 세트로 추출합니다. |
| **`create_val_severe.py`** | **악천후(Severe) 조건의 검증 데이터셋 생성** | 안개, 폭우, 혹은 심한 노이즈가 있는 악조건의 이미지를 검증 세트로 추출하여 모델의 강건성(Robustness)을 평가합니다. |

### 1.3. 실행 방법 및 결과

```bash
# 1. 원본 데이터를 COCO 포맷으로 변환
python pre_city_to_coco.py

# 2. 검증 세트 생성 (클리어 및 악천후 조건 분리)
python create_val_clear.py
python create_val_severe.py

# 실행 결과:
# - COCO 형식의 JSON 파일 (Annotations)
# - 원본 데이터가 전처리된 후 저장된 학습/검증 이미지 디렉토리