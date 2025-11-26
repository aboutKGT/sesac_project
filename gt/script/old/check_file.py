import os



def count_files_in_directory(path):
    """주어진 경로의 바로 아래 레벨에 있는 파일의 개수를 반환합니다."""
    
    # 1. 경로의 유효성 검사
    if not os.path.isdir(path):
        print(f"오류: '{path}'는 유효한 디렉터리가 아닙니다.")
        return 0
    
    file_count = 0
    
    # 2. 디렉터리 내용을 반복
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        
        # 3. 항목이 파일인지 확인
        if os.path.isfile(full_path):
            file_count += 1
            
    return file_count

val_path = '/home/elicer/dev/detectron2/final_data/val/images'
train_path = '/home/elicer/dev/detectron2/final_data/train/images'

# --- 사용 예시 ---
count = count_files_in_directory(train_path)

print(f"경로 '{train_path}' 바로 아래에 있는 파일의 개수: **{count}**개")