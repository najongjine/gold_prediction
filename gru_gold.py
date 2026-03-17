import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# ==========================================
# 스텝 1: yfinance에서 금값 데이터 가져오기 (10년치)
# ==========================================
print("yfinance에서 금값(GC=F) 데이터를 다운로드합니다...")
df = yf.download('GC=F', period='10y')

if isinstance(df.columns, pd.MultiIndex):
    df = df['Close'].copy()
    df.columns = ['Close']
else:
    df = df[['Close']].copy()

# ==========================================
# 스텝 2: 30일 이동평균 & ⭐️ 30일 뒤 수익률(Target) 계산
# ==========================================
# 일일 노이즈 제거를 위한 30일 평균선 적용
df['Smoothed_Close'] = df['Close'].rolling(window=30).mean()

# 스무딩된 가격을 바탕으로 일일 수익률 계산 (피처용)
df['Daily_Return'] = df['Smoothed_Close'].pct_change()

# ⭐️ 핵심: 정답지 만들기 (현재 스무딩 가격 대비 30일 뒤 스무딩 가격의 변동률)
# shift(-30)을 하면 30일 뒤의 미래 데이터를 현재 행으로 끌어올 수 있습니다.
df['Target_30d_Return'] = (df['Smoothed_Close'].shift(-30) - df['Smoothed_Close']) / df['Smoothed_Close']

# 학습용 데이터셋 생성 (최근 30일은 30일 뒤 미래 가격을 모르므로 NaN이 됨 -> 학습에서 제외)
df_train = df.dropna().copy()

# ==========================================
# 스텝 3: 데이터 스케일링
# ==========================================
scaler = MinMaxScaler()
# 피처(입력값)만 스케일링합니다. 정답(Target)은 어차피 퍼센트(%) 비율이므로 그대로 씁니다.
feature_cols = ['Smoothed_Close', 'Daily_Return']
scaled_features = scaler.fit_transform(df_train[feature_cols])

# 타겟값 분리
targets = df_train['Target_30d_Return'].values

# ==========================================
# 스텝 4: 윈도우 슬라이딩 (Window Sliding)
# ==========================================
def create_dataset_direct(features, targets, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size, :])
        # 윈도우의 가장 마지막 날에 해당하는 30일 뒤 수익률을 정답으로 지정
        y.append(targets[i + window_size - 1])
    return np.array(X), np.array(y)

WINDOW_SIZE = 60 # 과거 60일의 패턴을 보고 판단
X, y = create_dataset_direct(scaled_features, targets, WINDOW_SIZE)

# ==========================================
# 스텝 5: Train / Test 분할 및 DataLoader
# ==========================================
split_index = int(len(X) * 0.8)

X_train = torch.FloatTensor(X[:split_index]).to(device)
y_train = torch.FloatTensor(y[:split_index]).unsqueeze(1).to(device)
X_test = torch.FloatTensor(X[split_index:]).to(device)
y_test = torch.FloatTensor(y[split_index:]).unsqueeze(1).to(device)

batch_size = 32
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

# ==========================================
# 스텝 6: PyTorch GRU 모델 정의
# ==========================================
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :]) # 1개의 변동률 스칼라 값 출력
        return out

INPUT_DIM = 2 
HIDDEN_DIM = 64
OUTPUT_DIM = 1
NUM_LAYERS = 2

model = GRUModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==========================================
# 스텝 7: 학습 루프
# ==========================================
epochs = 50
print("\n--- 본격적인 학습을 시작합니다 ---")

for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_loss/len(train_loader):.6f} | Test Loss: {test_loss.item():.6f}')

print("학습 완료!")

# ==========================================
# 스텝 8: ⭐️ 향후 30일 뒤 다이렉트 예측 (실전)
# ==========================================
print("\n--- 현재 기준 30일 뒤 금값 변동률 예측 ---")

# 진짜 현재 상황을 알기 위해 dropna()를 하지 않은 원본 df에서 가장 최근 60일치를 가져옵니다.
# (최근 데이터는 Target_30d_Return이 NaN이겠지만, 모델 입력값인 Feature로는 문제없이 쓸 수 있습니다.)
latest_60_days = df[feature_cols].tail(WINDOW_SIZE).values

# 훈련 때 사용한 스케일러로 변환
latest_scaled = scaler.transform(latest_60_days)
current_seq = torch.FloatTensor(latest_scaled).unsqueeze(0).to(device)

model.eval()
with torch.no_grad():
    # 모델 출력값이 곧 30일 뒤의 예상 수익률(Return)입니다.
    predicted_return = model(current_seq).item()

# 현재 실제 가격 확인 (스무딩 기준)
current_smoothed_price = df['Smoothed_Close'].iloc[-1]
# 예상 가격 계산
predicted_future_price = current_smoothed_price * (1 + predicted_return)

# 결과 출력
print(f"현재 스무딩된 금값 (오늘): $ {current_smoothed_price:.2f}")
print("--------------------------------------------------")
if predicted_return > 0:
    print(f"📈 30일 뒤 예측: 현 시점 대비 **{predicted_return * 100:.2f}% 상승** 예상")
else:
    print(f"📉 30일 뒤 예측: 현 시점 대비 **{predicted_return * 100:.2f}% 하락** 예상")
print(f"🎯 30일 뒤 예상 가격: $ {predicted_future_price:.2f}")
print("--------------------------------------------------")