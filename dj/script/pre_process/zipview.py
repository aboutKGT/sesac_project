import zipfile
import cv2
import numpy as np
import json

current_index = 0
file_list = []
window_name = "Validation Viewer"

image_zip_path = "/home/elicer/data/090.승용_자율주행차_주간_자동차_전용도로_데이터/01-1.정식개방데이터/Validation/01.원천데이터/VS.zip"
label_zip_path = "/home/elicer/data/090.승용_자율주행차_주간_자동차_전용도로_데이터/01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip"

def build_file_list():
    global file_list
    with zipfile.ZipFile(image_zip_path, 'r') as zipf:
        file_list = [f for f in zipf.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        file_list.sort()

def get_image_and_label(index):
    img_name = file_list[index]
    image = None
    label_data = None

    with zipfile.ZipFile(image_zip_path, 'r') as zipf:
        img_bytes = zipf.read(img_name)
        img_array = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    json_name = img_name.rsplit('.',1)[0] + ".json"
    with zipfile.ZipFile(label_zip_path, 'r') as zipf:
        if json_name in zipf.namelist():
            json_bytes = zipf.read(json_name)
            label_data = json.loads(json_bytes.decode('utf-8'))

    return image, label_data

def mouse_callback(event, x, y, flags, param):
    global current_index
    if event == cv2.EVENT_LBUTTONDOWN:
        _, _, w, _ = cv2.getWindowImageRect(window_name)
        if x < w / 2:
            move_prev_image()
        else:
            move_next_image()

def move_next_image():
    global current_index
    if current_index + 1 < len(file_list):
        current_index += 1
        display_current_image()
    else:
        display_current_image(status_message="Last Image")

def move_prev_image():
    global current_index
    if current_index - 1 >= 0:
        current_index -= 1
        display_current_image()
    else:
        display_current_image(status_message="First Image")

def display_current_image(status_message=None):
    global current_index, file_list
    if not file_list:
        return

    img, _ = get_image_and_label(current_index)
    if img is None:
        print(f"이미지 로드 실패: {current_index}")
        return

    if status_message:
        cv2.putText(img, status_message, (50, img.shape[0]-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    cv2.imshow(window_name, img)

def main():
    global file_list, current_index
    build_file_list()
    if not file_list:
        print("VS.zip 내부에서 이미지 파일을 찾을 수 없습니다.")
        return

    display_current_image(status_message="First Image")

    print("A/Left: 이전, D/Right: 다음, Q/Esc: 종료")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key in [ord('a')]:
            move_prev_image()
        elif key in [ord('d')]:
            move_next_image()

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
