1. 프로젝트 개요
   목적: 과거 10년간의 금값 데이터를 학습하여, 현 시점 기준 30일 뒤의 금값 변동률(상승/하락폭)과 예상 가격을 예측하는 딥러닝 시계열 모델을 구축한다.

핵심 기술 (Tech Stack): \* Python, PyTorch (딥러닝 프레임워크)

yfinance (데이터 수집)

Pandas, NumPy (데이터 처리)

Scikit-learn (스케일링 및 시계열 교차 검증)

타겟 환경: 라이브 2D AI 에이전트의 '금값 분석 조언'을 위한 백엔드 예측 엔진.

2. 데이터 수집 및 전처리 (Data Pipeline)
   💡 설계 의도 (Human Learning): 주식이나 금값 데이터는 하루하루의 노이즈(변동성)가 너무 심해 AI가 추세를 읽기 어렵습니다. 따라서 30일 이동평균선(Smoothed)을 구하여 노이즈를 걷어내고, '절대적인 가격'이 아닌 '수익률(변동 비율)'을 학습하게 하여 모델의 안정성을 높입니다.

데이터 수집: yfinance를 사용하여 금 선물(GC=F)의 최근 10년치 종가(Close) 데이터를 가져온다.

파생 변수(Feature) 생성:

Smoothed_Close: 30일 이동평균선 (과거 30일의 평균).

Daily_Return: 일일 스무딩 종가 변동률 (Smoothed_Close의 pct_change()).

정답(Target) 라벨 생성:

Target_30d_Return: 현재 스무딩 가격 대비 30일 뒤의 수익률.

계산식: (30일 뒤 Smoothed_Close - 현재 Smoothed_Close) / 현재 Smoothed_Close

결측치 처리: 이동평균 및 타겟 계산 과정에서 발생하는 결측치(NaN) 행은 모두 제거(dropna)한다.

데이터 스케일링: MinMaxScaler를 사용해 입력 특성(Smoothed_Close, Daily_Return)을 0과 1 사이로 정규화한다.

3. 시계열 데이터 구조화 (Window Sliding)
   💡 설계 의도 (Human Learning): 시계열 딥러닝은 "어제 가격" 하나만 보고 내일을 맞추는 것이 아니라, "최근 N일간의 흐름"을 하나의 묶음으로 보고 패턴을 파악합니다. 이를 윈도우 슬라이딩 기법이라고 합니다.

윈도우 사이즈 (Window Size): 60일 (약 2~3개월 치의 거래일 흐름)

입력 데이터(X) 형태: (샘플 수, 60일, 2개 특성)의 3차원 배열.

출력 데이터(y) 형태: 입력된 60일 직후 시점의 Target_30d_Return (스칼라 값).

60일씩 창을 한 칸씩 이동해가며 전체 학습용 X, y 세트를 구축한다.

4. 인공지능 모델 아키텍처 (GRU Network)
   💡 설계 의도 (Human Learning): GRU(Gated Recurrent Unit)는 과거의 정보 중 '중요한 것은 남기고 불필요한 것은 잊어버리는' 능력을 가진 순환 신경망입니다. 긴 시계열 데이터를 처리할 때 LSTM보다 가벼우면서도 비슷한 성능을 냅니다.

모델 구조: PyTorch 기반 nn.Module 상속 클래스

Input Dimension: 2 (Smoothed_Close, Daily_Return)

Hidden Dimension: 64

Number of Layers: 2 (GRU 층을 2단으로 쌓음)

Output Layer: nn.Linear(64, 1) (마지막 시퀀스의 은닉 상태를 입력받아 최종 수익률 1개를 출력)

손실 함수 (Loss Function): MSE (Mean Squared Error)

옵티마이저 (Optimizer): Adam (학습률 lr=0.001)

5. 학습 및 검증 전략: 시계열 교차 검증 (TimeSeriesSplit)
   💡 설계 의도 (Human Learning): 일반적인 머신러닝처럼 데이터를 섞어서 검증하면 미래 데이터로 과거를 맞추는 '데이터 누수'가 발생합니다. 따라서 과거부터 시간 순서를 유지하며 학습 구간을 늘려가는 교차 검증으로 모델의 진짜 실력을 엄밀하게 테스트합니다.

검증 방식: sklearn.model_selection.TimeSeriesSplit (n_splits=5) 사용.

학습 하이퍼파라미터: Batch Size = 32, Epochs = 30.

독립적 학습: 각 Fold가 시작될 때마다 모델의 가중치와 옵티마이저를 완전히 새로 초기화한다. (이전 Fold의 학습 내용이 다음 Fold로 넘어가는 것을 방지).

평가: 각 Fold별 Test Loss(MSE)를 측정하고, 최종적으로 5개 Fold의 평균 Test Loss를 출력하여 모델 구조의 안정성을 확인한다.

6. 실전 투입 및 미래 예측 (Production & Inference)
   💡 설계 의도 (Human Learning): 성능 검증이 끝났다면, 내일 당장의 금값을 예측하기 위해 '가장 최근 데이터'까지 모두 포함하여 똑똑한 최종 모델을 만들어야 합니다.

최종 모델 전체 학습: 교차 검증 시 나누었던 Train/Test 구분 없이, 구축된 전체 X, y 데이터를 모두 사용하여 새로운 final_model을 다시 30 Epoch 동안 학습시킨다.

현시점 데이터 추출: 원본 데이터 프레임에서 가장 최근 60일 치의 특성 데이터를 가져와 스케일링 후 텐서로 변환한다.

미래 예측 (Inference): \* 학습 완료된 최종 모델에 최근 60일 데이터를 입력하여 predicted_return (30일 뒤 예상 수익률)을 도출한다.

결과 변환 및 출력:

현재 시점의 마지막 스무딩 종가 \* (1 + predicted_return) 계산을 통해 **30일 뒤 예상 금값(달러)**을 산출한다.

예측 결과가 양수면 상승, 음수면 하락으로 터미널에 명확히 포맷팅하여 출력한다.
