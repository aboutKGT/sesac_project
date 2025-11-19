import zipfile
import os
import sys
from collections import defaultdict

# ZIP 파일 경로 설정 (절대 경로 사용)
# IMAGE_ZIP_PATH = "/home/elicer/data/090.승용_자율주행차_주간_자동차_전용도로_데이터/01-1.정식개방데이터/Validation/01.원천데이터/VS.zip"
# LABEL_ZIP_PATH = "/home/elicer/data/090.승용_자율주행차_주간_자동차_전용도로_데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip"
IMAGE_ZIP_PATH = "/home/elicer/data/090.승용_자율주행차_주간_자동차_전용도로_데이터/01-1.정식개방데이터/Validation/01.원천데이터/VS.zip"
LABEL_ZIP_PATH = "/home/elicer/data/090.승용_자율주행차_주간_자동차_전용도로_데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip"

# [변경됨] 출력 파일 이름 및 경로 설정
OUTPUT_FILE_NAME_BASE = "dataset_tree_report.md"
# 저장 위치를 요청하신 최상위 경로로 지정
OUTPUT_DIR = "/home/elicer/data/090.승용_자율주행차_주간_자동차_전용도로_데이터/"
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME_BASE)

# 그림 문자 정의 (트리 출력용)
TREE_MID = "├── "
TREE_END = "└── "
TREE_VERT = "│   "
TREE_EMPTY = "    "

def get_zip_file_info(zip_path):
    """
    ZIP 파일 내부의 파일 목록과 크기 정보를 딕셔너리로 반환하고 진행 상황을 표시합니다.
    """
    file_info = {}
    print(f"\n파일 처리 중: {os.path.basename(zip_path)}")
    count = 0
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            namelist = zipf.namelist()
            for name in namelist:
                if not name.endswith('/'):
                    info = zipf.getinfo(name)
                    file_info[name] = info.file_size
                    
                    count += 1
                    if count % 1000 == 0:
                        sys.stdout.write('\n')
                    elif count % 100 == 0:
                        sys.stdout.write('|')
                    elif count % 10 == 0:
                        sys.stdout.write('.')
                    sys.stdout.flush() 
                    
        print(f"\n총 {count}개 파일 처리 완료.")
    except FileNotFoundError:
        print(f"오류: ZIP 파일을 찾을 수 없습니다: {zip_path}")
        return None, f"오류: ZIP 파일을 찾을 수 없습니다: {zip_path}"
    except zipfile.BadZipFile:
        print(f"오류: ZIP 파일이 손상되었습니다: {zip_path}")
        return None, f"오류: ZIP 파일이 손상되었습니다: {zip_path}"
    
    return file_info, None

def analyze_zip_summary(file_info_dict, zip_name):
    """ZIP 파일 전체의 통계를 분석하여 Markdown 문자열로 반환합니다."""
    total_files = len(file_info_dict)
    total_size_bytes = sum(file_info_dict.values())
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    all_folders = set(os.path.dirname(path) for path in file_info_dict.keys() if os.path.dirname(path))
    
    # [오류 수정 완료] 첫 번째 서브폴더 이름(문자열)만 추출하여 set에 저장
    first_level_folder_names = set()
    for path in all_folders:
        parts = path.split('/')
        if parts:
            first_level_folder_names.add(parts[0]) # 리스트(parts) 대신 첫 번째 문자열(parts[0]) 추가

    extensions = defaultdict(int)
    for path in file_info_dict.keys():
        _, ext = os.path.splitext(path)
        if ext: extensions[ext.lower()] += 1
            
    summary = f"## {zip_name} 전체 개요\n\n"
    summary += f"- **총 파일 개수:** {total_files} 개\n"
    summary += f"- **총 폴더 개수 (전체):** {len(all_folders) + 1} 개\n"
    summary += f"- **최상위 서브폴더 개수:** {len(first_level_folder_names)} 개\n"
    summary += f"- **전체 용량:** {total_size_mb:.2f} MB\n"
    summary += "- **파일 종류별 개수:**\n"
    for ext, count in sorted(extensions.items()):
        summary += f"  - `{ext}`: {count} 개\n"
    summary += "\n"
    return summary

def generate_folder_tree_with_glyphs(file_info_dict, max_depth=None):
    """
    그림 문자를 사용하여 파일이 존재하는 폴더 트리를 생성하고 통계를 추가합니다.
    max_depth가 지정되면 해당 깊이까지만 표시하며, 진행 상황을 출력합니다.
    """
    output_lines = []
    all_dirs_with_files = set()
    for path in file_info_dict.keys():
        dirname = os.path.dirname(path)
        if dirname:
            all_dirs_with_files.add(dirname)
            head = dirname
            while head and head != '/':
                all_dirs_with_files.add(head)
                head = os.path.dirname(head)
    
    sorted_dirs = sorted(list(all_dirs_with_files))

    # [오류 수정 완료] first_level_folders_count 계산 로직 수정
    first_level_folder_names = set()
    for path in all_dirs_with_files:
         parts = path.split('/')
         if parts:
             first_level_folder_names.add(parts[0]) # 리스트 대신 문자열 추가
    first_level_folders_count = len(first_level_folder_names)

    stats_summary_root = calculate_folder_stats("", file_info_dict)
    output_lines.append(f". ({first_level_folders_count}개 최상위 폴더) {stats_summary_root}")

    count = 0
    for i, folder_path in enumerate(sorted_dirs):
        # [통일된 진행 상황 표시]
        count += 1
        if count % 1000 == 0:
            sys.stdout.write('\n')
        elif count % 100 == 0:
            sys.stdout.write('|')
        elif count % 10 == 0:
            sys.stdout.write('.')
        sys.stdout.flush()
        
        depth = folder_path.count('/')
        if max_depth is not None and depth >= max_depth:
            continue
        
        is_last = True
        for next_path in sorted_dirs[i+1:]:
            if next_path.startswith(os.path.dirname(folder_path) + "/"):
                is_last = False
                break
        
        prefix = ""
        parts = folder_path.split('/')
        for d in range(depth):
            parent_path = "/".join(parts[:d+1])
            is_parent_last = True
            for next_path in sorted_dirs:
                if next_path.startswith(parent_path + "/") and next_path != parent_path:
                   is_parent_last = False
                   break
            prefix += TREE_EMPTY if is_parent_last else TREE_VERT
        
        glyph = TREE_END if is_last else TREE_MID
        stats_summary = calculate_folder_stats(folder_path, file_info_dict)
        output_lines.append(f"{prefix}{glyph}{os.path.basename(folder_path)}/ {stats_summary}")
        
    return "\n".join(output_lines) + "\n"

def calculate_folder_stats(folder_path, all_files_dict):
    """특정 폴더 경로의 통계 (총 폴더 수, 파일 수, 용량, 확장자 개수)를 계산합니다."""
    if folder_path == "":
        folder_files = list(all_files_dict.keys())
    else:
        folder_files = [p for p in all_files_dict.keys() if p.startswith(f"{folder_path}/")]
        
    count_files = len(folder_files)
    size_bytes = sum(all_files_dict[p] for p in folder_files)
    size_mb = size_bytes / (1024 * 1024)
    
    extensions = defaultdict(int)
    for path in folder_files:
        _, ext = os.path.splitext(path)
        if ext: extensions[ext.lower()] += 1

    ext_summary = ", ".join([f"{ext}: {count}개" for ext, count in sorted(extensions.items())])
    
    sub_folders = set()
    for path in folder_files:
        dirname = os.path.dirname(path)
        if dirname.startswith(folder_path) and dirname != folder_path:
            sub_folders.add(dirname)
    
    count_folders = len(sub_folders)

    return f"(폴더 {count_folders}개, 파일 {count_files}개, {size_mb:.2f} MB, {ext_summary})"


def main():
    summary_buffer = f"# 파일 구조 보고서\n\n최종 업데이트: {os.path.getmtime(__file__)}\n\n"
    tree_buffer = ""

    # 1. 이미지 ZIP 파일 처리
    print("--- 1단계: 원본 이미지 ZIP 파일 데이터 읽기 ---")
    image_files, err = get_zip_file_info(IMAGE_ZIP_PATH)
    if err:
        summary_buffer += f"## {os.path.basename(IMAGE_ZIP_PATH)}\n\n{err}\n\n"
    else:
        print(f"--- 2단계: {os.path.basename(IMAGE_ZIP_PATH)} 통계 분석 및 트리 생성 중 ---")
        summary_buffer += analyze_zip_summary(image_files, os.path.basename(IMAGE_ZIP_PATH))
        
        tree_buffer += f"## {os.path.basename(IMAGE_ZIP_PATH)} 2단계 폴더 트리 구조\n\n```\n"
        tree_buffer += generate_folder_tree_with_glyphs(image_files, max_depth=1)
        tree_buffer += "\n```\n\n"

        tree_buffer += f"## {os.path.basename(IMAGE_ZIP_PATH)} 전체 폴더 트리 구조\n\n```\n"
        tree_buffer += generate_folder_tree_with_glyphs(image_files)
        tree_buffer += "\n```\n\n"
        print("\n" * 2)

    # 2. 라벨 ZIP 파일 처리
    print("--- 3단계: 라벨 ZIP 파일 데이터 읽기 ---")
    label_files, err = get_zip_file_info(LABEL_ZIP_PATH)
    if err:
        summary_buffer += f"## {os.path.basename(LABEL_ZIP_PATH)}\n\n{err}\n\n"
    else:
        print(f"--- 4단계: {os.path.basename(LABEL_ZIP_PATH)} 통계 분석 및 트리 생성 중 ---")
        summary_buffer += analyze_zip_summary(label_files, os.path.basename(LABEL_ZIP_PATH))

        tree_buffer += f"## {os.path.basename(LABEL_ZIP_PATH)} 2단계 폴더 트리 구조\n\n```\n"
        tree_buffer += generate_folder_tree_with_glyphs(label_files, max_depth=1)
        tree_buffer += "\n```\n\n"

        tree_buffer += f"## {os.path.basename(LABEL_ZIP_PATH)} 전체 폴더 트리 구조\n\n```\n"
        tree_buffer += generate_folder_tree_with_glyphs(label_files)
        tree_buffer += "\n```\n\n"
        print("\n" * 2)

    output_buffer = summary_buffer + tree_buffer

    print(f"--- 5단계: 결과 파일을 {os.path.basename(OUTPUT_FILE_PATH)}로 저장 중 ---")
    try:
        # [변경됨] OUTPUT_FILE_PATH 변수 사용
        with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write(output_buffer)
        
        full_path = os.path.abspath(OUTPUT_FILE_PATH)
        print(f"\n성공적으로 파일 정보를 저장했습니다.")
        print(f"--> 저장 위치 (절대 경로): {full_path}")
        print("\n--- 모든 작업 완료 ---")

    except IOError as e:
        print(f"파일 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
