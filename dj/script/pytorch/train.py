import os  # 파일 시스템 작업을 위한 모듈
import subprocess  # 서브프로세스 실행을 위한 모듈
import numpy as np  # 수치 연산을 위한 모듈
import torch  # PyTorch 메인 모듈
import torch.nn as nn  # 신경망 레이어 모듈
from torch.utils.data import DataLoader  # 데이터 로더 모듈
from torch.utils.tensorboard import SummaryWriter  # TensorBoard 로깅 모듈
from torchvision import transforms, datasets  # 데이터 변환 및 데이터셋 모듈

# 실행 시 dj 브랜치로 고정
try:
    project_root = os.path.join(os.environ.get('HOME', os.path.expanduser('~')), 'dev')
    result = subprocess.run(['git', 'checkout', 'dj'], cwd=project_root, 
                          capture_output=True, text=True, timeout=2)
except:
    pass  # Git 명령 실패 시 무시

## 트레이닝 필요한 파라메터를 설정하기
# 홈 디렉토리 기준으로 프로젝트 루트 디렉토리 설정
home_dir = os.environ.get('HOME', os.path.expanduser('~'))  # 홈 디렉토리 가져오기
project_root = os.path.join(home_dir, 'dev', 'dj')  # 프로젝트 루트 (dev/dj)

lr = 1e-3  # 학습률 설정
batch_size = 64  # 배치 크기 설정
num_epoch=10  # 에폭 수 설정
data_dir = os.path.join(project_root, 'data/mnist')  # 데이터 저장 디렉토리
ckpt_dir = os.path.join(project_root, 'checkpoint')  # 체크포인트 저장 디렉토리
log_dir = os.path.join(project_root, 'log')  # 로그 저장 디렉토리
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 사용 가능 여부에 따라 디바이스 설정
class Net(nn.Module):  # 신경망 모델 클래스 정의
    def __init__(self):  # 초기화 함수
        super(Net, self).__init__()  # 부모 클래스 초기화
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=0, bias=True)  # 첫 번째 합성곱 레이어
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 첫 번째 맥스 풀링 레이어
        self.relu1 = nn.ReLU()  # 첫 번째 ReLU 활성화 함수
        
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=0, bias=True)  # 두 번째 합성곱 레이어
        self.drop2 = nn.Dropout2d(p=0.5)  # 두 번째 드롭아웃 레이어
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 두 번째 맥스 풀링 레이어
        self.relu2 = nn.ReLU()  # 두 번째 ReLU 활성화 함수
        
        self.fc1 = nn.Linear(in_features=320, out_features=50, bias=True)  # 첫 번째 완전 연결 레이어
        self.relu_fc1 = nn.ReLU()  # 완전 연결 레이어용 ReLU
        self.drop1_fc1 = nn.Dropout(p=0.5)  # 완전 연결 레이어용 드롭아웃
        
        self.fc2 = nn.Linear(in_features=50, out_features=10, bias=True)  # 두 번째 완전 연결 레이어 (출력 레이어)
    def forward(self,x):  # 순전파 함수
        x = self.conv1(x)  # 첫 번째 합성곱 연산
        x = self.pool1(x)  # 첫 번째 풀링 연산
        x = self.relu1(x)  # 첫 번째 ReLU 활성화
        
        x = self.conv2(x)  # 두 번째 합성곱 연산
        x = self.drop2(x)  # 두 번째 드롭아웃 적용
        x = self.pool2(x)  # 두 번째 풀링 연산
        x = self.relu2(x)  # 두 번째 ReLU 활성화
        
        x = x.view(x.size(0), -1)  # Flatten - 2D 텐서를 1D로 변환
        x = self.fc1(x)  # 첫 번째 완전 연결 레이어 통과
        x = self.relu_fc1(x)  # ReLU 활성화
        x = self.drop1_fc1(x)  # 드롭아웃 적용
        x = self.fc2(x)  # 출력 레이어 통과
        return x  # 최종 출력 반환
## 네트워크를 저장하거나 불러오는 함수
def save(ckpt_dir, net, optim, epoch):  # 모델 저장 함수
    if not os.path.exists(ckpt_dir):  # 체크포인트 디렉토리가 없으면
        os.makedirs(ckpt_dir)  # 디렉토리 생성
        
    torch.save({'net':net.state_dict(), 'optim':optim.state_dict()},  # 네트워크와 옵티마이저 상태 저장
               os.path.join(ckpt_dir, 'model_epoch%d.pth' % epoch))  # 파일 경로 지정 (절대 경로)
def load(ckpt_dir, net, optim):  # 모델 불러오기 함수
    ckpt_lst = os.listdir(ckpt_dir)  # 체크포인트 디렉토리의 파일 목록 가져오기
    ckpt_lst.sort()  # 파일 목록 정렬
    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]))  # 가장 최근 체크포인트 로드 (절대 경로)
    
    net.load_state_dict(dict_model['net'])  # 네트워크 가중치 로드
    optim.load_state_dict(dict_model['optim'])  # 옵티마이저 상태 로드
    return net, optim  # 로드된 네트워크와 옵티마이저 반환
## MNIST 데이터 불러오기
print("Loading MNIST dataset...")  # 데이터 로딩 시작 메시지
transform = transforms.Compose([  # 데이터 변환 파이프라인 정의
    transforms.ToTensor(),  # 이미지를 텐서로 변환
    transforms.Normalize((0.5,), (0.5,))  # 정규화 (평균 0.5, 표준편차 0.5)
])

dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)  # MNIST 학습 데이터셋 로드
print(f"MNIST dataset loaded successfully! Total samples: {len(dataset)}")  # 데이터 로딩 완료 메시지

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 데이터 로더 생성

num_data = len(loader.dataset)  # 전체 데이터 개수
num_batch = np.ceil(num_data/batch_size)  # 배치 개수 계산

## 네트워크 설정 및 필요한 손실함수 구하기
net = Net().to(device)  # 네트워크를 디바이스로 이동
params = net.parameters()  # 네트워크 파라미터 가져오기
fn_loss = nn.CrossEntropyLoss().to(device)  # 손실 함수 정의 (Cross Entropy Loss)
fn_pred = lambda output: torch.softmax(output, dim=1)  # 예측 확률 계산 함수 (softmax)
fn_acc = lambda pred, label: (pred.argmax(dim=1) == label).float().mean()  # 정확도 계산 함수
optim = torch.optim.Adam(params, lr=lr)  # Adam 옵티마이저 생성
writer = SummaryWriter(log_dir)  # TensorBoard writer 생성
## 학습 루프
for epoch in range(num_epoch+1):  # 에폭 반복
    net.train()  # 학습 모드로 설정
    loss_arr = []  # 손실값 저장 리스트
    acc_arr = []  # 정확도 저장 리스트

    for batch, (input, label) in enumerate(loader,1):  # 배치 반복
        input = input.to(device)  # 입력 데이터를 디바이스로 이동
        label = label.to(device)  # 레이블을 디바이스로 이동
        
        output = net(input)  # 순전파 - 네트워크에 입력 전달
        pred = fn_pred(output)  # 예측 확률 계산
        acc = fn_acc(pred, label)  # 정확도 계산
        
        optim.zero_grad()  # 옵티마이저 그래디언트 초기화
        
        loss = fn_loss(output, label)  # 손실 계산
        loss.backward()  # 역전파 - 그래디언트 계산
        optim.step()  # 옵티마이저 스텝 - 파라미터 업데이트
        
        loss_arr += [loss.item()]  # 손실값 저장
        acc_arr += [acc.item()]  # 정확도 저장       

    # 에폭 종료 후 평균 손실과 정확도 출력
    avg_loss = np.mean(loss_arr)  # 평균 손실 계산
    avg_acc = np.mean(acc_arr)  # 평균 정확도 계산
    print('Epoch: {}/{}\tLoss: {:.6f}\tAccuracy: {:.4f}'.format(  # 에폭별 학습 결과 출력
        epoch, num_epoch, avg_loss, avg_acc))  # 에폭 번호, 평균 손실, 평균 정확도
    
    writer.add_scalar('Loss/train', avg_loss, epoch)  # TensorBoard에 손실 기록
    writer.add_scalar('Accuracy/train', avg_acc, epoch)  # TensorBoard에 정확도 기록
    
    save(ckpt_dir, net, optim, epoch)  # 체크포인트 저장
    writer.close()  # TensorBoard writer 닫기
print('Training Complete!')  # 학습 완료 메시지 출력
