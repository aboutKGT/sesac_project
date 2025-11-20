import os

# 1. 경로 설정 (사용자가 제공한 경로)
img_dir = '/home/elicer/dev/gt/data/new_data/val_image'
lbl_dir = '/home/elicer/dev/gt/data/new_data/test_label'

# 2. 이미지 폴더의 파일 목록을 집합(set)으로 저장 (검색 속도 향상을 위해)
try:
    valid_images = set(os.listdir(img_dir))
    print(f"기준 이미지 로드 완료: {len(valid_images)}개")
except FileNotFoundError:
    print(f"오류: 이미지 경로를 찾을 수 없습니다. -> {img_dir}")
    exit()

# 3. 라벨 폴더 순회 및 삭제 작업
deleted_count = 0
skipped_count = 0

if os.path.exists(lbl_dir):
    print("라벨 정리 시작...")
    for lbl_file in os.listdir(lbl_dir):
        # .json 파일인 경우에만 로직 수행
        if lbl_file.endswith('.json'):
            # 예: CK_..._F.png.json -> CK_..._F.png (.json 제거)
            target_img_name = lbl_file[:-5] 

            # 해당 이미지 이름이 이미지 폴더 리스트에 없는 경우 삭제
            if target_img_name not in valid_images:
                file_path = os.path.join(lbl_dir, lbl_file)
                try:
                    os.remove(file_path)
                    print(f"[삭제] {lbl_file} (이미지 없음)")
                    deleted_count += 1
                except OSError as e:
                    print(f"[에러] {lbl_file} 삭제 실패: {e}")
            else:
                # 이미지가 짝이 맞게 있는 경우
                skipped_count += 1
    
    print("-" * 30)
    print(f"작업 완료.")
    print(f"삭제된 라벨 파일: {deleted_count}개")
    print(f"유지된 라벨 파일: {skipped_count}개")

else:
    print(f"오류: 라벨 경로를 찾을 수 없습니다. -> {lbl_dir}")