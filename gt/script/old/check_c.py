import json
from collections import Counter

# 생성된 COCO 포맷 JSON 경로
JSON_PATH = '/home/elicer/dev/gt/data/label_data/train_coco_full.json'

def check_stats():
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # 카테고리 ID와 이름 매핑
    id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    # 어노테이션 개수 세기
    cnt = Counter()
    for ann in data['annotations']:
        cat_id = ann['category_id']
        cnt[id_to_name[cat_id]] += 1
        
    print(f"--- {JSON_PATH} 통계 ---")
    print(f"총 이미지 수: {len(data['images'])}")
    print(f"총 어노테이션 수: {len(data['annotations'])}")
    print("\n[클래스별 개수]")
    for name, count in cnt.most_common():
        print(f"{name}: {count}")

    if cnt['freespace'] == 0:
        print("\n[!!!경고!!!] 'freespace' 클래스의 데이터가 0개입니다. 전처리 코드를 수정해야 합니다.")
    else:
        print(f"\n'freespace' 데이터가 {cnt['freespace']}개 존재합니다.")

if __name__ == "__main__":
    check_stats()