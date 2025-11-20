import json
import random
import os
import shutil
import glob

# annotations_label = '/home/elicer/data/090.승용_자율주행차_주간_자동차_전용도로_데이터/vs_sample/labels/coco_annotations.json'

# with open(annotations_label, "r") as f:
#     coco = json.load(f)

# images = coco["images"]
# annotations = coco["annotations"]

# random.seed(42)

# image_ids = [img["id"] for img in images]
# random.shuffle(image_ids)

# train_size = int(len(image_ids)*0.8)
# train_ids = set(image_ids[:train_size])
# val_ids = set(image_ids[train_size:])

# train_images = [img for img in images if img["id"] in train_ids]
# val_images = [img for img in images if img["id"] in val_ids]

# train_annotations = [ann for ann in annotations if ann['image_id'] in train_ids]
# val_annotations = [ann for ann in annotations if ann['image_id'] in val_ids]

# categories = coco["categories"]

# train_json = {"images": train_images, "annotations": train_annotations, "categories": categories}
# val_json = {"images": val_images, "annotations": val_annotations, "categories": categories}

# with open("train.json", "w") as f:
#     json.dump(train_json, f)

# with open("val.json", "w") as f:
#     json.dump(val_json, f)

# ########################## train_json, val_json과 동일하게 이미지 파일 분류 ##########################

# img_dir = "/home/elicer/data/090.승용_자율주행차_주간_자동차_전용도로_데이터/vs_sample/images"
# train_dir = "/home/elicer/dev/ss/script/train_images"
# val_dir = "/home/elicer/dev/ss/script/val_images"

# os.makedirs(train_dir, exist_ok=True)
# os.makedirs(val_dir, exist_ok=True)


# with open("train.json", "r") as f:
#     train_json = json.load(f)
# with open("val.json", "r") as f:
#     val_json = json.load(f)

# def find_image_file(base_name, img_dir):
#     base_name2 = base_name.replace('.jpg', '')
#     pattern = os.path.join(img_dir, f"*{base_name2}*.jpg")
#     matches = glob.glob(pattern)
#     if len(matches) > 0:
#         return matches[0]
#     else:
#         return None
# count = 0
# for img in train_json['images']:
#     base_name = img['file_name']
#     src_file = find_image_file(base_name, img_dir)
#     if src_file:
#         shutil.copy(src_file, os.path.join(train_dir, os.path.basename(src_file)))
#     else:
#         print(f"⚠️ 파일 없음: {base_name}")
#         count += 1

# count2 = 0
# for img in val_json["images"]:
#     base_name = img['file_name']
#     src_file = find_image_file(base_name, img_dir)
#     if src_file:
#         shutil.copy(src_file, os.path.join(val_dir, os.path.basename(src_file)))    
#     else:
#         print(f"⚠️ 파일 없음: {base_name}")
#         count2 += 1

# print(f"count갯수 : {count}, count2갯수 : {count2}")



################ 이미지 파일 이름과 json 파일 file_name 이름 똑같이 맞추기 ################

train_json_path = "/home/elicer/dev/ss/script/train.json"
val_json_path = "/home/elicer/dev/ss/script/val.json"
train_dir = "/home/elicer/dev/ss/script/train_images/"
val_dir = "/home/elicer/dev/ss/script/val_images/"

def rename_images(json_path, img_dir):
    with open(json_path, "r") as f:
        data = json.load(f)

    for img in data["images"]:
        base_name = img["file_name"]  # JSON에 기록된 이름
        # 최종적으로 맞춰야 할 파일명 (확장자 붙여줌)
        new_name = base_name
        # 현재 폴더 안에서 해당 이미지 ID에 맞는 파일 찾기
        # (이미 분류된 상태라면 폴더 안에 하나씩 존재한다고 가정)
        candidates = [f for f in os.listdir(img_dir) if str(img["id"]) in f or base_name in f]

        if candidates:
            old_path = os.path.join(img_dir, candidates[0])
            new_path = os.path.join(img_dir, new_name)
            os.rename(old_path, new_path)
            print(f"✅ {candidates[0]} → {new_name}")
        else:
            print(f"⚠️ {base_name} 에 해당하는 파일을 찾지 못했습니다.")

# 실행
rename_images(train_json_path, train_dir)
rename_images(val_json_path, val_dir)