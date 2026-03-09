import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import platform

# 1. 조회 기간 설정: 오늘 기준 10년 전부터 오늘까지
end = datetime.datetime.today()
try:
    start = end.replace(year=end.year - 10)
except ValueError:
    # 오늘이 2월 29일(윤일)인 경우 10년 전은 2월 28일로 처리
    start = end.replace(year=end.year - 10, day=28)

# 2. yfinance 데이터 가져오기
yf_tickers = {
    'DX-Y.NYB': 'Dollar_Index',
    '^TNX': 'US_10Y_Yield',
    '^VIX': 'VIX',
    'BTC-USD': 'Bitcoin'
}

print(f"yfinance 데이터 다운로드 중... ({start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')})")
try:
    # yf.download로 한 번에 가져오고 'Close' 컬럼만 추출
    combined_data = yf.download(list(yf_tickers.keys()), start=start, end=end)['Close']
    
    # 컬럼명 변경 (보기 쉽게 변환)
    combined_data.rename(columns=yf_tickers, inplace=True)
    
    # yfinance 최신 버전에서 다중 인덱스가 생성되는 경우 평탄화
    if isinstance(combined_data.columns, pd.MultiIndex):
        combined_data.columns = combined_data.columns.get_level_values(0)
        
    print("yfinance 데이터 로드 성공!")
except Exception as e:
    print(f"yfinance 데이터 로드 중 오류 발생: {e}")
    combined_data = pd.DataFrame()

# 3. 데이터 전처리
print("\n데이터 전처리 중...")
if not combined_data.empty:
    # 주말/휴일 등 시장이 열리지 않아 생긴 빈칸(NaN)을 앞의 값으로 채움
    combined_data.ffill(inplace=True) 
    
    # 최초 데이터 이전의 빈칸은 뒤의 값으로 채움 (10년치 데이터가 안 되는 종목 처리용)
    combined_data.bfill(inplace=True)

    print("\n최종 데이터 상위 5개 행:")
    print(combined_data.head())

    # 4. 데이터 시각화
    if platform.system() == 'Windows':
        print("\nWindows 환경이 감지되었습니다. 그래프를 출력합니다.")
        
        # 달러 인덱스와 10년물 금리를 비교 시각화
        if 'Dollar_Index' in combined_data.columns and 'US_10Y_Yield' in combined_data.columns:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # 달러 인덱스 (왼쪽 y축)
            color = 'tab:blue'
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Dollar Index', color=color)
            ax1.plot(combined_data.index, combined_data['Dollar_Index'], color=color, label='Dollar Index')
            ax1.tick_params(axis='y', labelcolor=color)

            # 10년물 국채 금리 (오른쪽 y축)
            ax2 = ax1.twinx()  
            color = 'tab:red'
            ax2.set_ylabel('US 10Y Yield (%)', color=color)
            ax2.plot(combined_data.index, combined_data['US_10Y_Yield'], color=color, label='10Y Yield')
            ax2.tick_params(axis='y', labelcolor=color)

            plt.title('Dollar Index vs US 10Y Treasury Yield (10 Years)')
            fig.tight_layout() 
            plt.show()
        else:
            print("시각화에 필요한 데이터가 누락되었습니다.")
    else:
        print(f"\n현재 운영체제는 {platform.system()}입니다. 시각화를 건너뜁니다.")
else:
    print("다운로드된 데이터가 없어 진행할 수 없습니다.")