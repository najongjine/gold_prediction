import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import platform
import time

# 1. 조회 기간 설정
start = datetime.datetime(2020, 1, 1)
end = datetime.datetime.today()

# 2. yfinance 데이터 가져오기 (S&P500 제외)
yf_tickers = {
    'DX-Y.NYB': 'Dollar_Index',
    '^TNX': 'US_10Y_Yield',
    '^VIX': 'VIX',
    'BTC-USD': 'Bitcoin'
}

print("yfinance 데이터 다운로드 중...")
yf_data = pd.DataFrame()
try:
    yf_data = yf.download(list(yf_tickers.keys()), start=start, end=end)['Close']
    yf_data.rename(columns=yf_tickers, inplace=True)
    print("yfinance 데이터 로드 성공!")
except Exception as e:
    print(f"yfinance 데이터 로드 중 오류 발생: {e}")

# 3. FRED 데이터 가져오기 (재시도 로직 제거)
fred_tickers = {
    'CPIAUCSL': 'US_CPI',
    'DFII10': 'US_10Y_TIPS'
}

print("\nFRED 데이터 다운로드 중 (개별 다운로드)...")
fred_data_list = []

for ticker, name in fred_tickers.items():
    try:
        temp_df = web.DataReader(ticker, 'fred', start, end)
        temp_df.rename(columns={ticker: name}, inplace=True)
        fred_data_list.append(temp_df)
        print(f"FRED 데이터 로드 성공: {name} ({ticker})")
    except Exception as e:
        print(f"FRED 데이터 로드 실패 ({name}): {e}")
    
    time.sleep(1)

if fred_data_list:
    fred_data = pd.concat(fred_data_list, axis=1)
else:
    fred_data = pd.DataFrame()

# 4. 전체 데이터 병합
print("\n전체 데이터 병합 중...")
if yf_data.empty and fred_data.empty:
    print("다운로드된 데이터가 없어 병합할 수 없습니다.")
else:
    combined_data = pd.concat([yf_data, fred_data], axis=1)

    # 빈칸 채우기
    combined_data.ffill(inplace=True) 
    combined_data.bfill(inplace=True)

    print("\n병합된 데이터 상위 5개 행:")
    print(combined_data.head())

    # 5. 데이터 시각화
    if platform.system() == 'Windows':
        print("\nWindows 환경이 감지되었습니다. 그래프를 출력합니다.")
        
        # S&P500 대신 달러 인덱스와 10년물 금리를 비교 시각화
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

            plt.title('Dollar Index vs US 10Y Treasury Yield')
            fig.tight_layout() 
            plt.show()
        else:
            print("시각화에 필요한 데이터가 누락되었습니다.")
    else:
        print(f"\n현재 운영체제는 {platform.system()}입니다. 시각화를 건너뜁니다.")