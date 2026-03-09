import pandas as pd
import pandas_datareader.data as web
import datetime
import matplotlib.pyplot as plt
import platform  # OS 확인을 위한 모듈 추가

# 1. 조회할 기간 설정: 오늘 기준 10년 전부터 현재까지
end = datetime.datetime.today()
try:
    start = end.replace(year=end.year - 10)
except ValueError:
    # 오늘이 2월 29일(윤일)일 경우 10년 전 날짜를 2월 28일로 처리
    start = end.replace(year=end.year - 10, day=28)

# 2. FRED에서 데이터 불러오기
# 티커 'IPG334S': Industrial Production: Manufacturing: Durable Goods: Computer and Electronic Product
try:
    print(f"FRED 데이터 다운로드 중... ({start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')})")
    tech_production = web.DataReader('IPG334S', 'fred', start, end)
    
    # 데이터 확인
    print("데이터 로드 완료. 상위 5개 행:")
    print(tech_production.head())
    
    # 결측치 제거
    tech_production = tech_production.dropna()

    # 3. 데이터 시각화 (Windows 환경일 때만 실행)
    current_os = platform.system()
    
    if current_os == 'Windows':
        print("\nWindows 환경이 감지되었습니다. 그래프를 화면에 출력합니다.")
        plt.figure(figsize=(12, 6))
        plt.plot(tech_production.index, tech_production['IPG334S'], label='Computer & Electronic Product Index', color='dodgerblue')
        
        plt.title('Industrial Production: Computer & Electronic Product (Last 10 Years)')
        plt.xlabel('Date')
        plt.ylabel('Index (2017=100)') 
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        # Windows가 아닐 경우의 처리 (예: Linux, Darwin 등)
        print(f"\n현재 운영체제는 {current_os}입니다. 화면 시각화를 건너뜁니다.")

except Exception as e:
    print(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")