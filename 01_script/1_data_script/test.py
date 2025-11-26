import json
import numpy as np
import cv2
import os

# =========================================================
# [설정] 테스트할 파일 경로
# =========================================================
# 2D 라벨 (Target)
FILE_2D = '/home/elicer/data/092_AD_City_Day/Validation/02_label_data/lable_day_clear/2D/CK_A01_R01_day_clear_01008464_F.png.json'

# 3D 라벨 (Source) - 경로가 3D인지 3D_json인지 확인 필요 (사용자 입력 기준)
FILE_3D = '/home/elicer/data/092_AD_City_Day/Validation/02_label_data/lable_day_clear/3D/LK_A01_R01_day_clear_01008464.json'

# 캘리브레이션 (K 차량용 사용 - LK 파일이므로)
CALIB_FILE = '/home/elicer/data/092_AD_City_Day/K_CF-L_calib.txt'

# =========================================================
# [함수] 캘리브레이션 및 투영 (역행렬 적용됨)
# =========================================================
def parse_calib(calib_path):
    print(f"🔍 캘리브레이션 로드: {calib_path}")
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    k_mat, rt_mat = None, None
    for i, line in enumerate(lines):
        if "CameraExtrinsicMat" in line:
            vals = [float(x) for x in lines[i+1].strip().split(',')]
            rt_mat = np.array(vals).reshape(4, 4)
            # [중요] 역행렬 적용 Check
            try: 
                rt_mat = np.linalg.inv(rt_mat)
                print("   -> Extrinsic Matrix 역행렬(Inverse) 적용 완료")
            except: 
                print("   -> 역행렬 적용 실패 (Singular Matrix)")
        if "CameraMat" in line:
            vals = [float(x) for x in lines[i+1].strip().split(',')]
            k_mat = np.array(vals).reshape(3, 3)
    return k_mat, rt_mat

def project_3d_to_2d(points_3d, k_mat, rt_mat):
    if len(points_3d) == 0: return None
    n = len(points_3d)
    pts_homo = np.hstack((np.array(points_3d), np.ones((n, 1))))
    
    # 투영: World -> Camera
    pts_cam = rt_mat @ pts_homo.T 
    
    # 투영: Camera -> Image
    pts_img_homo = k_mat @ pts_cam[:3, :]
    
    # Z값(깊이) 확인용 로그
    z_vals = pts_img_homo[2, :]
    # 1.0m 앞보다 더 멀리 있는 점만 유효 (Safety Guard)
    valid_indices = z_vals > 1.0 
    
    if np.sum(valid_indices) < 3: 
        return None
    
    pts_img_homo = pts_img_homo[:, valid_indices]
    u = pts_img_homo[0, :] / pts_img_homo[2, :]
    v = pts_img_homo[1, :] / pts_img_homo[2, :]
    
    return np.vstack((u, v)).T.astype(np.int32)

def calculate_iou(boxA, boxB):
    # box: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea

# =========================================================
# [메인] 테스트 실행
# =========================================================
def main():
    # 1. 데이터 로드
    if not os.path.exists(FILE_2D) or not os.path.exists(FILE_3D):
        print("❌ 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
        return

    k_mat, rt_mat = parse_calib(CALIB_FILE)
    with open(FILE_2D, 'r') as f: data_2d = json.load(f)
    with open(FILE_3D, 'r') as f: data_3d = json.load(f)

    print(f"\n🚀 [Test Start] 2D vs 3D IoU Matching")
    print(f"   - 2D Objects: {len(data_2d['annotations'])}개")
    print(f"   - 3D Objects: {len(data_3d['annotations'])}개")

    # 2. 3D 객체 투영 및 BBox 생성
    projected_objs = []
    print("\n🔹 [Step 1] 3D 객체 투영 결과:")
    for obj3d in data_3d['annotations']:
        cls = obj3d.get('class', 'Unknown')
        if '3D_points' in obj3d:
            poly = project_3d_to_2d(obj3d['3D_points'], k_mat, rt_mat)
            
            if poly is not None:
                x, y, w, h = cv2.boundingRect(poly)
                dist = obj3d.get('distance', -1)
                print(f"   [3D] {cls:<15} (Dist: {dist:.2f}m) -> BBox: [{x}, {y}, {w}, {h}]")
                
                # 유효성 체크 (이미지 범위 1920x1080 + Margin)
                if -500 < x < 2500 and -500 < y < 1500:
                    projected_objs.append({'bbox': [x, y, w, h], 'class': cls, 'dist': dist})
                else:
                    print(f"        ⚠️ 박스가 너무 큽니다/벗어났습니다 (Filter Out)")
            else:
                # print(f"   [3D] {cls} -> 투영 실패 (Camera 뒤쪽)")
                pass

    # 3. 2D 객체와 매칭 시도
    print("\n🔹 [Step 2] 2D 객체와 IoU 비교:")
    matched_count = 0
    
    for obj2d in data_2d['annotations']:
        cat_id = obj2d['category_id']
        # 카테고리 이름 찾기
        cat_name = next((c['name'] for c in data_2d['category'] if c['id'] == cat_id), str(cat_id))
        bbox_2d = obj2d['bbox'] # [x, y, w, h]
        
        print(f"   👉 [2D] {cat_name:<15} BBox: {bbox_2d}")
        
        best_iou = 0
        best_match = None
        
        for p_obj in projected_objs:
            iou = calculate_iou(bbox_2d, p_obj['bbox'])
            
            # IoU가 0.01이라도 있으면 로그 출력 (매칭 가능성 확인용)
            if iou > 0.01:
                print(f"        vs 3D {p_obj['class']:<10} (IoU: {iou:.4f}) - Dist: {p_obj['dist']:.2f}m")
            
            if iou > best_iou:
                best_iou = iou
                best_match = p_obj
        
        if best_iou > 0.3: # 임계값 테스트
            print(f"        ✅ MATCHED! Distance Updated: {best_match['dist']:.2f}m")
            matched_count += 1
        else:
            print(f"        ❌ No Match (Best IoU: {best_iou:.4f})")

    print(f"\n🏁 [Result] 총 {len(data_2d['annotations'])}개 중 {matched_count}개 매칭 성공")

if __name__ == "__main__":
    main()