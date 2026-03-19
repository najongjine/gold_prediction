import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

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
# 스텝 2: 60일 이동평균 & 60일 뒤 수익률(Target) 계산
# ==========================================
df['Smoothed_Close'] = df['Close'].rolling(window=60).mean()
df['Daily_Return'] = df['Smoothed_Close'].pct_change()
df['Target_60d_Return'] = (df['Smoothed_Close'].shift(-60) - df['Smoothed_Close']) / df['Smoothed_Close']

df_train = df.dropna().copy()

# ==========================================
# 스텝 3: 데이터 스케일링
# ==========================================
scaler = MinMaxScaler()
feature_cols = ['Smoothed_Close', 'Daily_Return']
scaled_features = scaler.fit_transform(df_train[feature_cols])
targets = df_train['Target_60d_Return'].values

# ==========================================
# 스텝 4: 윈도우 슬라이딩 (Window Sliding)
# ==========================================
def create_dataset_direct(features, targets, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i : i + window_size, :])
        y.append(targets[i + window_size - 1])
    return np.array(X), np.array(y)

WINDOW_SIZE = 120 
X, y = create_dataset_direct(scaled_features, targets, WINDOW_SIZE)

# ==========================================
# 스텝 5: PyTorch GRU 모델 정의
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
        out = self.fc(out[:, -1, :]) 
        return out

INPUT_DIM = 2 
HIDDEN_DIM = 64
OUTPUT_DIM = 1
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 30 # CV 과정이 길어질 수 있어 에폭을 30으로 조정

# ==========================================
# 스텝 6: TimeSeriesSplit 교차 검증
# ==========================================
tscv = TimeSeriesSplit(n_splits=5)
fold_results = []

print("\n--- TimeSeriesSplit 교차 검증 시작 (5 Folds) ---")

for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
    print(f"\n[Fold {fold}] 학습 데이터: {len(train_index)}개, 테스트 데이터: {len(test_index)}개")
    
    # 1. Fold별 데이터 분할 및 텐서 변환
    X_train_fold = torch.FloatTensor(X[train_index]).to(device)
    y_train_fold = torch.FloatTensor(y[train_index]).unsqueeze(1).to(device)
    X_test_fold = torch.FloatTensor(X[test_index]).to(device)
    y_test_fold = torch.FloatTensor(y[test_index]).unsqueeze(1).to(device)
    
    train_dataset = TensorDataset(X_train_fold, y_train_fold)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Fold마다 모델과 옵티마이저를 새로 초기화 (데이터 누수 방지)
    model_cv = GRUModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_cv.parameters(), lr=0.001)
    
    # 3. Fold 학습
    model_cv.train()
    for epoch in range(EPOCHS):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model_cv(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # 4. Fold 평가
    model_cv.eval()
    with torch.no_grad():
        test_predictions = model_cv(X_test_fold)
        test_loss = criterion(test_predictions, y_test_fold).item()
        
    print(f"✅ Fold {fold} Test Loss: {test_loss:.6f}")
    fold_results.append(test_loss)

print("\n=====================================")
print(f"📊 평균 Test Loss: {np.mean(fold_results):.6f}")
print("=====================================")

# ==========================================
# 스텝 7: 전체 데이터로 최종 모델 학습 (실전용)
# ==========================================
print("\n--- 실전 예측을 위한 전체 데이터 최종 학습 ---")

X_full = torch.FloatTensor(X).to(device)
y_full = torch.FloatTensor(y).unsqueeze(1).to(device)

full_dataset = TensorDataset(X_full, y_full)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=False)

final_model = GRUModel(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, NUM_LAYERS).to(device)
final_optimizer = optim.Adam(final_model.parameters(), lr=0.001)

final_model.train()
for epoch in range(EPOCHS):
    for batch_X, batch_y in full_loader:
        final_optimizer.zero_grad()
        outputs = final_model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        final_optimizer.step()

print("최종 학습 완료!")

# ==========================================
# 스텝 8: 향후 60일 뒤 다이렉트 예측
# ==========================================
print("\n--- 현재 기준 60일 뒤 금값 변동률 예측 ---")

latest_60_days = df[feature_cols].tail(WINDOW_SIZE).values
latest_scaled = scaler.transform(latest_60_days)
current_seq = torch.FloatTensor(latest_scaled).unsqueeze(0).to(device)

final_model.eval()
with torch.no_grad():
    predicted_return = final_model(current_seq).item()

current_smoothed_price = df['Smoothed_Close'].iloc[-1]
predicted_future_price = current_smoothed_price * (1 + predicted_return)

print(f"현재 스무딩된 금값 (최근 60일 평균): $ {current_smoothed_price:.2f}")
print("--------------------------------------------------")
if predicted_return > 0:
    print(f"📈 60일 뒤 예측: 현 시점 대비 **{predicted_return * 100:.2f}% 상승** 예상")
else:
    print(f"📉 60일 뒤 예측: 현 시점 대비 **{predicted_return * 100:.2f}% 하락** 예상")
print(f"🎯 60일 뒤 예상 가격: $ {predicted_future_price:.2f}")
print("--------------------------------------------------")
