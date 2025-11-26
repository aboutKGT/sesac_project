import cv2
import os

# 모델 호출 파일 import
from inference_utils import MyPredictor as load_model 

# [설정] 100번 학습한 멍청한 모델 경로
# (아까 학습 로그에 찍힌 OUTPUT_DIR 경로의 model_final.pth)
MODEL_PATH = "/home/elicer/dev/gt/custom_model/20251120_062503/model_final.pth" # <<< 현재는 아무 모델이나 지정해둠
TEST_IMAGE_PATH = "/home/elicer/dev/detectron2/final_data/val/images/08_174514_221206_07.jpg" # 아무거나 하나

OUTPUT_PATH = "/home/elicer/dev/02_output"

def data_load():
    pass

def main():
    
    # 1. 모델 로딩 (여기서 시간이 좀 걸림 - 딱 한 번만 실행됨)
    print("모델을 로드하는 중입니다...")
    # 멍청한 모델이므로 임계값(thresh)을 0.05로 아주 낮게 설정
    predictor = load_model(weight_path=MODEL_PATH, score_thresh=0.05)
    print("모델 로드 완료!")

    # 2. 이미지 읽기
    if not os.path.exists(TEST_IMAGE_PATH):
        print("테스트 이미지가 없습니다. 경로를 확인해주세요.")
        return

    img = cv2.imread(TEST_IMAGE_PATH)
    
    # 3. 추론 실행 (함수 호출 한 방으로 끝!)
    result_img = predictor.run(img)
    
    # 4. 결과 저장 및 확인
    save_path = OUTPUT_PATH + "/inference_result.jpg"
    cv2.imwrite(save_path, result_img)
    print(f"추론 완료! 결과가 저장되었습니다: {save_path}")

if __name__ == "__main__":
    main()