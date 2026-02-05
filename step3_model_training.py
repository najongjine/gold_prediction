import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 전처리된 데이터 로드
df = pd.read_csv('processed_gold_data.csv')

# 2. 학습/테스트 데이터 분리 (시계열 데이터이므로 순서 유지)
X = df.drop('Target', axis=1)
y = df['Target']

# 시계열 특성을 고려하여 뒤쪽 20%를 테스트셋으로 사용
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# 3. LightGBM 모델 설정 및 학습
# 하이퍼파라미터는 일반적인 시계열용으로 초기 설정
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# 4. 예측 및 평가
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\n[3단계 모델 평가]")
print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# 5. 결과 시각화
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual 20d Return', color='green', alpha=0.7)
plt.plot(y_pred, label='Predicted 20d Return', color='red', linestyle='--')
plt.title('Gold 20-day Return Prediction (LightGBM)')
plt.legend()
plt.savefig('step3_prediction_result.png')

# 6. 피처 중요도 시각화
plt.figure(figsize=(10, 6))
lgb.plot_importance(model, max_num_features=10)
plt.title('Feature Importance')
plt.savefig('step3_feature_importance.png')

print("\n결과 파일 저장 완료: step3_prediction_result.png, step3_feature_importance.png")
