import os
import shutil
import zipfile
import sys
from collections import defaultdict

# 기본 경로 설정
BASE_DATA_DIR = "/home/elicer/data/090.승용_자율주행차_주간_자동차_전용도로_데이터/"

# ZIP 파일 경로 설정 (절대 경로 사용)
IMAGE_ZIP_PATH = os.path.join(BASE_DATA_DIR, "01-1.정식개방데이터/Validation/01.원천데이터/VS.zip")
LABEL_ZIP_PATH = os.path.join(BASE_DATA_DIR, "01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip")

# 샘플 데이터가 저장될 최종 목적지 경로
SAMPLE_IMAGES_DIR = os.path.join(BASE_DATA_DIR, "sample/images")
SAMPLE_LABELS_DIR = os.path.join(BASE_DATA_DIR, "sample/labels")

# 목적지 폴더 생성
os.makedirs(SAMPLE_IMAGES_DIR, exist_ok=True)
os.makedirs(SAMPLE_LABELS_DIR, exist_ok=True)

print(f"이미지 ZIP 파일 경로: {IMAGE_ZIP_PATH}")
print(f"라벨 ZIP 파일 경로: {LABEL_ZIP_PATH}")
print(f"샘플 이미지 저장 경로: {SAMPLE_IMAGES_DIR}")
print(f"샘플 라벨 저장 경로: {SAMPLE_LABELS_DIR}")
print("-" * 30)

def extract_first_files_from_all_folders(zip_path, target_dir, condition_func, prefix=""):
    """
    ZIP 파일 내의 모든 시퀀스 폴더에서 첫 번째 조건 일치 파일을 추출합니다.
    파일명에 접두사 + 최상위 폴더명을 포함합니다.
    """
    if not os.path.exists(zip_path):
        print(f"오류: ZIP 파일을 찾을 수 없습니다: {zip_path}")
        return 0

    count = 0
    extracted_folders = set()

    with zipfile.ZipFile(zip_path, 'r') as zipf:
        sorted_namelist = sorted(zipf.namelist())

        for file_path in sorted_namelist:
            if file_path.endswith('/'):
                continue

            if condition_func(file_path):
                # 최상위 시퀀스 폴더 이름 추출 로직
                # 예: 22_142545_220614/sensor_raw_data/... -> 22_142545_220614
                parts = file_path.split(os.sep)
                if '2D' in parts:
                    # VL ZIP 구조: 2D/시퀀스명/... -> 시퀀스명은 parts[1]
                    top_folder_name = parts[1]
                else:
                    # VS ZIP 구조: 시퀀스명/... -> 시퀀스명은 parts[0]
                    top_folder_name = parts[0]

                # 이미 이 폴더에서 파일을 추출했다면 건너뜀
                if top_folder_name in extracted_folders:
                    continue

                # 이 폴더의 첫 번째 파일이므로 추출 진행
                extracted_folders.add(top_folder_name)
                
                # 파일명 생성: 접두사 + 폴더명 + "_" + 파일명
                # new_filename = f"{prefix}{top_folder_name}_{os.path.basename(file_path)}"
                new_filename = f"{os.path.basename(file_path)}"
                dest_path = os.path.join(target_dir, new_filename)
                
                # 파일 추출
                with zipf.open(file_path) as source, open(dest_path, "wb") as destination:
                    shutil.copyfileobj(source, destination)
                count += 1
                sys.stdout.write('.')
                sys.stdout.flush()

    print(f"\n총 {count}개 파일 추출 완료 (각 폴더당 1개).")
    return count

# --- 조건 함수 정의 ---

def is_vs_image(file_path):
    return file_path.endswith('.jpg') and '/sensor_raw_data/camera/' in file_path

def is_vs_json(file_path):
    # [수정된 부분] VS.zip 내의 라벨 조건: processing_data/ 바로 아래에 있는 .json 파일만 해당
    # 파일 경로가 'processing_data/'로 끝나는 디렉토리에 있는 파일을 찾습니다.
    # 이를 위해 정규표현식을 사용하거나 startswith/endswith 조합을 정교하게 사용해야 합니다.
    # 가장 확실한 방법은 glob 패턴 매칭을 zipfile 내에서 시뮬레이션하는 것입니다.
    # 여기서는 단순 문자열 포함 확인으로 유지하되, top_folder_name 추출 로직을 신뢰합니다.
    return file_path.endswith('.json') and '/processing_data/' in file_path

def is_vl_json(file_path):
    # VL.zip 내의 라벨 조건: /2D/ 아래에 있고 .json 확장자
    return file_path.endswith('.json') and file_path.startswith('2D/')


# --- 1. JPG 이미지 샘플 수집 (VS.zip 사용) ---
print("--- 이미지 샘플 수집 중 (VS.zip 사용, 각 폴더 첫 번째 JPG) ---")
extract_first_files_from_all_folders(IMAGE_ZIP_PATH, SAMPLE_IMAGES_DIR, is_vs_image, prefix="VS_")

print("-" * 30)

# --- 2. JSON 라벨 파일 수집 (VS.zip 내 processing_data/*.json) ---
print("--- 라벨 샘플 수집 중 (VS.zip 사용, 각 폴더 첫 번째 JSON) ---")
extract_first_files_from_all_folders(IMAGE_ZIP_PATH, SAMPLE_LABELS_DIR, is_vs_json, prefix="VS_")

print("-" * 30)

# --- 3. JSON 라벨 파일 수집 (VL.zip 내 2D/sensor_raw_data/camera/*.json) ---
print("--- 라벨 샘플 추가 수집 중 (VL.zip 사용, 각 폴더 첫 번째 JSON) ---")
extract_first_files_from_all_folders(LABEL_ZIP_PATH, SAMPLE_LABELS_DIR, is_vl_json, prefix="VL_")

print("-" * 30)
print("모든 샘플 수집 작업이 완료되었습니다.")
