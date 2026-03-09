import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 한글 폰트 설정 (Windows 환경)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("--- 2020년부터 현재까지의 금 가격 데이터 수집 및 EDA ---")
    
    # 금 선물(Gold Futures) 티커: 'GC=F'
    ticker = 'GC=F'
    start_date = '2020-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"[{ticker}] 데이터를 {start_date}부터 {end_date}까지 다운로드합니다...")
    gold_data = yf.download(ticker, start=start_date, end=end_date)
    
    if gold_data.empty:
        print("데이터를 가져오는 데 실패했습니다.")
        return

    print("\n1. 데이터 기본 정보 (최근 5일)")
    print(gold_data.tail())
    
    print("\n2. 데이터 결측치 확인")
    print(gold_data.isnull().sum())
    
    print("\n3. 데이터 요약 통계량")
    print(gold_data.describe())

    # 종가(Close) 기준으로 시각화 준비 (yfinance 버전에 따른 MultiIndex 처리)
    if isinstance(gold_data.columns, pd.MultiIndex):
        close_price = gold_data['Close'][ticker].dropna()
    else:
        close_price = gold_data['Close'].dropna()

    # --- EDA 시각화 ---
    plt.figure(figsize=(14, 10))
    
    # 1. 시계열 선 차트 및 단순/지수평활 이동평균선
    plt.subplot(2, 1, 1)
    plt.plot(close_price.index, close_price, label='종가 (Close)', color='orange', linewidth=2)
    
    # 단순 이동평균(SMA) 계산
    ma50 = close_price.rolling(window=50).mean()
    ma200 = close_price.rolling(window=200).mean()
    
    # 지수평활 이동평균(EMA) 계산
    ema50 = close_price.ewm(span=50, adjust=False).mean()
    ema200 = close_price.ewm(span=200, adjust=False).mean()
    
    # 이동평균선 플롯 (단순)
    plt.plot(close_price.index, ma50, label='50일 단순이동평균(SMA)', color='blue', linestyle='--', alpha=0.4)
    plt.plot(close_price.index, ma200, label='200일 단순이동평균(SMA)', color='red', linestyle='--', alpha=0.4)
    
    # 지수평활선 플롯 (지수평활 EMA)
    plt.plot(close_price.index, ema50, label='50일 지수평활(EMA)', color='darkblue', linestyle='-', alpha=0.8)
    plt.plot(close_price.index, ema200, label='200일 지수평활(EMA)', color='darkred', linestyle='-', alpha=0.8)
    
    plt.title('2020년 ~ 현재 금 가격 (Gold Futures, GC=F) 추이 및 지수평활 이동평균선')
    plt.xlabel('날짜')
    plt.ylabel('가격 (USD)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()

    # 2. 연도별 분포 (Boxplot)
    plt.subplot(2, 1, 2)
    df_plot = close_price.reset_index()
    df_plot.columns = ['Date', 'Price']
    df_plot['Year'] = df_plot['Date'].dt.year
    
    sns.boxplot(x='Year', y='Price', data=df_plot, hue='Year', palette='Set3', legend=False)
    plt.title('연도별 금 가격 분포 (Boxplot)')
    plt.xlabel('연도')
    plt.ylabel('가격 (USD)')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
