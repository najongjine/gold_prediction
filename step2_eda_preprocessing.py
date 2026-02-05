import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. 데이터 로드
gold = yf.download('GC=F', start='2015-01-01')
df = gold[['Close', 'High', 'Low', 'Open', 'Volume']].copy()

# 2. 데이터 전처리 및 파생변수 생성 (Feature Engineering)
# (1) 수익률 변수 (Log Returns)
df['Return_1d'] = np.log(df['Close'] / df['Close'].shift(1))
df['Return_5d'] = np.log(df['Close'] / df['Close'].shift(5))   # 1주일
df['Return_20d'] = np.log(df['Close'] / df['Close'].shift(20)) # 1개월
df['Return_60d'] = np.log(df['Close'] / df['Close'].shift(60)) # 1분기 (장기 추세)

# (2) 이동평균 및 이격도
df['MA20'] = df['Close'].rolling(window=20).mean()
df['MA60'] = df['Close'].rolling(window=60).mean()
df['Disparity20'] = (df['Close'] / df['MA20']) * 100

# (3) 변동성 (Volatility)
df['Vol_20d'] = df['Return_1d'].rolling(window=20).std()

# (4) RSI (상대강도지수) - 14일 기준
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# (5) 타겟 설정: '20일 후의 장기 수익률' 예측 (수업용 핵심)
df['Target_Next20d'] = df['Return_20d'].shift(-20)

df.dropna(inplace=True)

# 3. EDA - 상관관계 분석
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('eda_correlation.png')

# 4. PCA (주성분 분석)
# 피처만 추출 (Target 제외)
features = ['Return_1d', 'Return_5d', 'Return_20d', 'Return_60d', 'MA20', 'MA60', 'Vol_20d', 'RSI', 'Disparity20']
x = df[features]
x_scaled = StandardScaler().fit_transform(x)

pca = PCA(n_components=min(len(features), 5))
pca_result = pca.fit_transform(x_scaled)

print(f"PCA 설명 가능 분산 비율: {pca.explained_variance_ratio_}")
print(f"총 설명 가능 분산: {sum(pca.explained_variance_ratio_):.4f}")

# PCA 결과 데이터프레임 저장
df_pca = pd.DataFrame(data=pca_result, columns=[f'PC{i+1}' for i in range(pca.n_components)])
df_pca['Target'] = df['Target_Next20d'].values
df_pca.to_csv('processed_gold_data.csv', index=False)

# 5. 시각화 - 장기 수익률 분포
plt.figure(figsize=(10, 6))
sns.histplot(df['Target_Next20d'], kde=True, color='green')
plt.title('Distribution of 20-day Future Returns')
plt.savefig('eda_target_dist.png')

print("\n[2단계 완료]")
print("1. 파생변수 생성: 1d, 5d, 20d, 60d 수익률 및 RSI, 변동성 등")
print("2. PCA를 통해 데이터 압축 및 특성 추출 완료")
print("3. 'processed_gold_data.csv'에 전처리된 데이터 저장 완료 (LightGBM용)")
