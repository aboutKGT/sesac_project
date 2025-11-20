import json
from collections import Counter

# 방금 업그레이드해서 만든 JSON 파일 경로
JSON_PATH = '/home/elicer/dev/gt/data/label_data/train_coco_format.json'

def check_stats():
    print(f"파일 읽는 중... {JSON_PATH}")
    try:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("오류: 파일을 찾을 수 없습니다. 변환 스크립트를 먼저 실행했는지 확인해주세요.")
        return

    # 카테고리 ID와 이름 매핑
    id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # 어노테이션 개수 세기
    cnt = Counter()
    for ann in data['annotations']:
        cat_id = ann['category_id']
        if cat_id in id_to_name:
            cnt[id_to_name[cat_id]] += 1
        
    print("-" * 30)
    print(f"[데이터 통계] 총 이미지: {len(data['images'])}장 / 어노테이션: {len(data['annotations'])}개")
    print("-" * 30)
    
    # 결과 출력
    print(f"{'클래스명':<20} | {'개수':<10}")
    print("-" * 30)
    
    # Freespace와 TrafficSign이 잘 들어왔는지 확인
    found_freespace = False
    
    for name, count in cnt.most_common():
        print(f"{name:<20} | {count:<10}")
        if name == "Freespace":
            found_freespace = True

    print("-" * 30)
    if found_freespace and cnt["Freespace"] > 0:
        print("✅ 성공! 'Freespace' 데이터가 정상적으로 포함되었습니다.")
        print("   이제 학습을 시작하셔도 좋습니다!")
    else:
        print("❌ 경고: 'Freespace' 데이터가 여전히 0개입니다.")
        print("   변환 코드를 다시 실행했는지(python convert_to_coco.py) 확인해주세요.")

if __name__ == "__main__":
    check_stats()