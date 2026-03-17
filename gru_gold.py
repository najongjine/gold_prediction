import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# ==========================================
# 스텝 1: yfinance에서 금값 데이터 가져오기 (10년치)
# ==========================================
print("yfinance에서 금값(GC=F) 데이터를 다운로드합니다...")
df = yf.download('GC=F', period='10y')

# 종가(Close)만 추출
# yfinance 최신 버전 다중 인덱스 방지용 처리
if isinstance(df.columns, pd.MultiIndex):
    df = df['Close'].copy()  # 이미 데이터프레임이므로 to_frame() 제거
    df.columns = ['Close']   # 컬럼명을 'Close'로 통일
else:
    df = df[['Close']].copy()

# ==========================================
# 스텝 2: 30일 이동평균 (Smoothing) & 파생변수
# ==========================================
# 일일 노이즈 제거를 위한 30일 평균선 적용
df['Smoothed_Close'] = df['Close'].rolling(window=30).mean()

# 스무딩된 가격을 바탕으로 수익률(Return) 계산
df['Return'] = df['Smoothed_Close'].pct_change()

# NaN 데이터 날리기 (초기 30일치 데이터 삭제됨)
df = df.dropna()

# 모델에 넣을 피처(Feature) 선택
features = df[['Smoothed_Close', 'Return']].values

# ==========================================
# 스텝 3: 데이터 스케일링
# ==========================================
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(features)

# ==========================================
# 스텝 4: 윈도우 슬라이딩 (Window Sliding)
# ==========================================
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :])
        # 정답(y)은 다음 날의 스무딩된 가격(0번째 인덱스)
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

WINDOW_SIZE = 60 # 과거 60일(약 2달)의 패턴을 보고 다음을 예측
X, y = create_dataset(scaled_data, WINDOW_SIZE)

# ==========================================
# 스텝 5: Train / Test 분할 및 PyTorch 텐서 변환
# ==========================================
split_index = int(len(X) * 0.8)

# numpy 배열을 PyTorch Tensor로 변환하고 디바이스에 올림
X_train = torch.FloatTensor(X[:split_index]).to(device)
y_train = torch.FloatTensor(y[:split_index]).unsqueeze(1).to(device)
X_test = torch.FloatTensor(X[split_index:]).to(device)
y_test = torch.FloatTensor(y[split_index:]).unsqueeze(1).to(device)

# DataLoader 생성 (배치 학습용)
batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # 시계열이므로 shuffle=False 권장

# ==========================================
# 스텝 6: PyTorch GRU 모델 정의
# ==========================================
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # batch_first=True 로 설정하여 입력 형태를 (batch, seq, feature)로 맞춤
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 초기 은닉 상태(hidden state) 0으로 초기화
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        
        # GRU 층 통과
        out, _ = self.gru(x, h0)
        
        # 마지막 타임 스텝의 출력만 사용해서 예측
        out = self.fc(out[:, -1, :]) 
        return out

# 파라미터 세팅
INPUT_DIM = 2  # 피처 개수 (Smoothed_Close, Return)
HIDDEN_DIM = 64
OUTPUT_DIM = 1
NUM_LAYERS = 2

model = GRUModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)

# 손실 함수와 옵티마이저
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 스텝 7: 찐 학습 루프 (Training Loop)
# ==========================================
epochs = 50
print("\n--- 본격적인 학습을 시작합니다 ---")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()      # 1. 기울기 초기화
        outputs = model(batch_X)   # 2. 예측 (Forward)
        loss = criterion(outputs, batch_y) # 3. 오차 계산
        loss.backward()            # 4. 역전파 (Backward)
        optimizer.step()           # 5. 가중치 업데이트
        
        epoch_loss += loss.item()
        
    # 10 에포크마다 진행 상황 출력
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_loss/len(train_loader):.6f} | Test Loss: {test_loss.item():.6f}')

print("학습 완료!")

import matplotlib.pyplot as plt

print("\n--- 미래 30일 예측을 시작합니다 (버그 수정 버전) ---")

# 1. 가장 최근 60일치 데이터 가져오기 (초기 입력값)
last_60_days = scaled_data[-WINDOW_SIZE:]
current_seq = torch.FloatTensor(last_60_days).unsqueeze(0).to(device) # 형태: (1, 60, 2)

predicted_real_prices = []
model.eval()

# ⭐️ 원본 데이터의 가장 마지막 '실제' 가격 가져오기 (진짜 수익률 계산용)
last_real_price = df['Smoothed_Close'].iloc[-1]

with torch.no_grad():
    for i in range(30):
        # 1. 다음 날 가격 예측 (0~1 사이 스케일된 값 1개)
        pred_scaled_price = model(current_seq).item()
        
        # 2. 스케일링 원상복구 (실제 달러 가격으로 변환)
        # scaler.inverse_transform은 피처 2개(가격, 수익률)를 요구하므로 임시 배열 생성
        dummy_input = np.array([[pred_scaled_price, 0]]) 
        real_pred_price = scaler.inverse_transform(dummy_input)[0, 0]
        
        # 3. 진짜 달러 가격을 바탕으로 '진짜 수익률(Return)' 계산
        real_return = (real_pred_price - last_real_price) / last_real_price
        
        # 4. 예측된 실제 가격 저장
        predicted_real_prices.append(real_pred_price)
        
        # 5. 다음 스텝 입력을 위해 (예측가격, 진짜수익률)을 다시 스케일링
        new_features_scaled = scaler.transform(np.array([[real_pred_price, real_return]]))
        
        # 6. 윈도우 슬라이딩 (새로운 데이터 추가, 가장 오래된 1일치 밀어내기)
        new_step = torch.FloatTensor([[new_features_scaled[0]]]).to(device)
        current_seq = torch.cat((current_seq[:, 1:, :], new_step), dim=1)
        
        # 다음 루프를 위해 마지막 실제 가격을 오늘 예측한 가격으로 업데이트
        last_real_price = real_pred_price

# 3. 예측 결과 출력
print("\n[ 향후 30일 금값(스무딩 기준) 예측 결과 ]")
for i, price in enumerate(predicted_real_prices):
    print(f"Day {i+1}: $ {price:.2f}")

# ==========================================
# 📊 차트 시각화 (최근 100일 + 미래 30일)
# ==========================================
actual_prices = df['Smoothed_Close'].values[-100:]

plt.figure(figsize=(12, 6))
plt.plot(range(100), actual_prices, label='Actual Smoothed Price (Last 100 Days)', color='blue')
# 미래 30일은 100번째 인덱스(마지막 날짜)부터 자연스럽게 이어지도록 x좌표 설정
plt.plot(range(99, 99 + 30), predicted_real_prices, label='Predicted Price (Next 30 Days)', color='red', linestyle='dashed')

plt.title('Gold Price Prediction (GRU) - Bug Fixed')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()