import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# ==========================================
# 0. yfinance로 금 가격 데이터 불러오기 (최근 10년)
# ==========================================
# 오늘 날짜와 10년 전 날짜 계산
today = datetime.today()
ten_years_ago = today - relativedelta(years=10)

start_date = ten_years_ago.strftime('%Y-%m-%d')
end_date = today.strftime('%Y-%m-%d')

print(f"금 가격 데이터를 다운로드 중입니다... ({start_date} ~ {end_date})")

# period='max' 대신 start와 end 파라미터를 사용합니다.
raw_df = yf.download('GC=F', start=start_date, end=end_date)

# 최신 yfinance 버전에 맞춘 종가('Close') 추출 로직
close_data = raw_df['Close']

# 만약 데이터가 1차원(Series)이라면 2차원으로 변환
if isinstance(close_data, pd.Series):
    df = close_data.to_frame(name='Close')
# 이미 2차원(DataFrame)이라면 그대로 복사 후 컬럼명만 깔끔하게 'Close'로 통일
else:
    df = close_data.copy()
    df.columns = ['Close'] 

print(f"총 {len(df)}일치의 데이터를 성공적으로 불러왔습니다.")

# ==========================================
# 1. 결측치 채우기 & 2. 이상치 처리
# ==========================================
df['Close'] = df['Close'].ffill()

lower_bound = df['Close'].quantile(0.01)
upper_bound = df['Close'].quantile(0.99)
df['Close'] = df['Close'].clip(lower=lower_bound, upper=upper_bound)

# ==========================================
# 3. 기술적 지표 생성 (Feature Engineering)
# ==========================================
# 🚨 [수정] 절대가격은 모델에 혼동을 주므로, 모두 '비율'로 변환합니다.

# 3-1. 30일 지수평활선 이격도 (EMA 절대가격이 아닌 현재가와의 비율)
ema_30 = df['Close'].ewm(span=30, adjust=False).mean()
df['EMA_30_Ratio'] = df['Close'] / ema_30

# 3-2. 이격도 (현재가 / 20일 단순이동평균)
sma_20 = df['Close'].rolling(window=20).mean()
df['Disparity'] = df['Close'] / sma_20

# 3-3. RSI (14일) - 0~100 사이의 비율이므로 그대로 사용
df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()

# 3-4. MACD - 절대가격에 비례하므로 수익률 데이터로 계산
macd = MACD(close=df['Close'].pct_change() * 100) # 퍼센트 스케일로 변환
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()

# 3-5. 변동성 지표 (볼린저 밴드 폭 비율)
bollinger = BollingerBands(close=df['Close'], window=20, window_dev=2)
# 밴드 폭 역시 가격에 비례하므로 이동평균으로 나누어 비율로 만듭니다.
df['BB_Width_Ratio'] = bollinger.bollinger_wband() / sma_20

# ==========================================
# 4. 정상성 변환 및 5. 파생 변수 (Lag) 생성
# ==========================================
df['Return'] = df['Close'].pct_change()

for i in range(1, 4):
    df[f'Return_Lag_{i}'] = df['Return'].shift(i)

# ==========================================
# 6. 타겟 변수 수정 (회귀 예측 🎯)
# ==========================================
# '30일 이동평균선(추세)'을 기준으로 설정합니다.
trend_price = df['Close'].rolling(window=30).mean()

# 현재 시점의 추세(trend_price)와 30일 후의 추세를 비교
future_trend = trend_price.shift(-30)

# 30일 후의 추세가 현재 추세 대비 몇 퍼센트(%) 오르거나 내리는지를 계산합니다.
df['Target_Percentage'] = (future_trend - trend_price) / trend_price * 100
df.loc[future_trend.isna(), 'Target_Percentage'] = np.nan

# ==========================================
# 7. 실전 예측용 데이터 분리
# ==========================================
train_df = df.dropna(subset=['Target_Percentage'])
today_data = df.iloc[[-1]].copy()

# 🚨 [중요] 절대가격인 'Close'는 훈련 데이터에서 완벽히 삭제합니다.
cols_to_drop = ['Close', 'Target_Percentage']
X = train_df.drop(columns=cols_to_drop)
y = train_df['Target_Percentage']
X_today = today_data.drop(columns=cols_to_drop)

# ==========================================
# 8. 학습용(Train)과 테스트용(Test) 데이터 분할
# ==========================================
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

# ==========================================
# 9. LightGBM 모델 선언 및 학습
# ==========================================
print("\n[LightGBM 모델 학습을 시작합니다 (목표: 향후 30일 후 가격 추세 등락 퍼센트 예측)...]")
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 분류(Classifier) 모델이 아닌, 연속적인 수치를 예측하는 회귀(Regressor) 모델을 사용합니다.
model = lgb.LGBMRegressor(
    n_estimators=100,
    learning_rate=0.05,
    random_state=42,
    max_depth=5
)
model.fit(X_train, y_train)

# ==========================================
# 10. 모델 성능 평가
# ==========================================
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

# 직관적인 이해를 돕기 위한 방향성 적중률 (정확도) 계산
# 실제값과 예측값의 부호(방향)가 같으면 정답으로 간주
direction_correct = (np.sign(y_test) == np.sign(predictions)).sum()
accuracy_like = (direction_correct / len(y_test)) * 100

print(f"\n✅ 테스트 데이터(최근 2년) 예측 성능 평가:")
print(f" - MAE (평균 절대 오차): {mae:.3f}% (예측이 실제 등락률과 평균적으로 얼마나 차이나는지)")
print(f" - RMSE (평균 제곱근 오차): {rmse:.3f}%")
print(f" - R² (결정 계수): {r2:.3f}")
print(f" - 방향 적중률 (Accuracy): {accuracy_like:.1f}% (상승/하락 방향을 맞춘 비율)")

# ==========================================
# 11. 🔮 실전 예측
# ==========================================
future_pred = model.predict(X_today)[0]

print("\n=============================================")
if future_pred > 0:
    print(f"🔮 예측 결과: 오늘 기점 30일 후 금 가격 추세는 현재 대비 약 **[{future_pred:.2f}% 상승]**할 것으로 예측됩니다!")
else:
    print(f"🔮 예측 결과: 오늘 기점 30일 후 금 가격 추세는 현재 대비 약 **[{abs(future_pred):.2f}% 하락]**할 것으로 예측됩니다!")
print("=============================================")


import platform
import matplotlib.pyplot as plt

# ==========================================
# 12. 피처 중요도(Feature Importance) 시각화 (Windows 전용)
# ==========================================
# 현재 OS가 Windows인지 확인합니다.
if platform.system() == 'Windows':
    print("\n[Windows 환경 감지됨] 피처 중요도(Feature Importance) 그래프를 준비합니다...")
    
    # 윈도우 환경 matplotlib 한글 폰트(맑은 고딕) 깨짐 방지 및 마이너스 부호 설정
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
    
    # 그래프 캔버스 설정
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # LightGBM 내장 plot_importance 함수를 사용하여 상위 10개 피처 출력
    lgb.plot_importance(
        model, 
        ax=ax, 
        importance_type='split',  # 트리 분기에 해당 변수가 몇 번이나 사용되었는지 기준
        max_num_features=10,      # 상위 10개만 깔끔하게 출력
        height=0.6,
        title='금 가격 30일 추세 백분율 변화 예측 - 결정적 파생 변수 TOP 10',
        xlabel='중요도 (노드 분기에 사용된 횟수)',
        ylabel='파생 변수 (Features)'
    )
    
    plt.tight_layout()
    plt.show()  # 윈도우 환경에서만 팝업 창으로 그래프 출력
else:
    pass
