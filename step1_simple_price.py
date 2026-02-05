import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. 데이터 수집 (금 선물: GC=F)
# yfinance 최신 버전 대응을 위해 auto_adjust=True 옵션 고려
gold = yf.download('GC=F', start='2020-01-01')
data = gold[['Close']].copy()

# 2. 아주 단순한 피처 생성: '어제 가격'으로 '오늘 가격' 예측하기
data['Yesterday'] = data['Close'].shift(1)
data.dropna(inplace=True)

# 3. 데이터 분할
X = data[['Yesterday']]
y = data['Close']
split = int(len(data) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. 초간단 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 5. 예측
predictions = model.predict(X_test)

# 6. 결과 시각화 (서버 환경 고려하여 파일로 저장)
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual Price', color='gold')
plt.plot(y_test.index, predictions, label='Predicted Price', color='blue', linestyle='--')
plt.title('Gold Price Prediction - Step 1 (Simple Linear Regression)')
plt.legend()
plt.savefig('step1_result.png')
print("결과 그래프가 'step1_result.png'로 저장되었습니다.")

# 성능 지표 출력 (얼마나 개판인지 확인용)
from sklearn.metrics import mean_squared_error, r2_score
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")
print("\n[분석] R2 점수는 높게 나올 수 있으나, 그래프를 보면 예측이 실제 가격을 하루씩 '복사'해서 따라가는 것을 볼 수 있습니다.")
