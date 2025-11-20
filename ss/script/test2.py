MODEL_PATH = "/home/elicer/dev/gt/custom_model/20251120_062503/model_final.pth"
TEST_IMAGE_PATH = "/home/elicer/dev/detectron2/final_data/val/images/08_174514_221206_07.jpg" 
OUTPUT_PATH = "/home/elicer/dev/02_output"

def load_video_frames(video_path, resize=None, skip=1):

    # 1. video_path에 있는 영상 불러오기
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened(): 
        raise FileNotFoundError(f"비디오를 열 수 없습니다: {video_path}")
    
    frame_id = 0

    while True:
        # ret : True or False / frame : 현재프레임
        ret, frame = cap.read()
        if not ret:
            break
                    # skip이 1로 정해져있어 모든 프레임을 다 가져옴
        if frame_id % skip == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            yield frame_id, frame

        frame_id += 1

    cap.release()

def inference_video(predictor, frames, output_path, fps=None):

    # 메모리에 모든 프레임을 쌓지 않고 바로 영상으로 저장
    frame_id, first_frame = next(frames)
    h, w, _ = first_frame.shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    result_img = predictor.run(first_frame)

    for frame_id, frame in frames:
        result_img = predictor.run(frame)
        writer.write(result_img)

    writer.release()
    print(f"🎬 영상 저장 완료: {output_path}")
    

def main():
    
    print("모델을 로드하는 중입니다...")
    predictor = load_model(weight_path=MODEL_PATH, score_thresh=0.05)
    print("모델 로드 완료!")


    video_path = TEST_IMAGE_PATH  # 불러올 영상 경로 (현재는 jpg지만 실제 영상 정해지면 변경)
    save_path = OUTPUT_PATH + "/inference_result.mp4"


            # 영상 데이터 > 프레임 단위로 분할
    frames = load_video_frames(video_path)
    #frames에는 제너레이터 객체 반환 (반복문에서 하나씩 꺼내야됨)

    inference_video(predictor, frames, save_path)

if __name__ == "__main__":
    main()


