# pip install duckduckgo-search
from duckduckgo_search import DDGS

def get_recent_news(query, max_results=5):
    # DDGS 객체 생성
    with DDGS() as ddgs:
        print(f"[{query}] 최신 뉴스를 검색합니다...\n")
        
        # news() 함수로 뉴스 탭 검색 결과만 깔끔하게 가져오기
        # timelimit: "d"(하루), "w"(일주일), "m"(한달) 단위로 최신 데이터 필터링 가능
        results = ddgs.news(
            keywords=query,
            region="kr-kr",    # 한국어 결과 우선 (글로벌 영문 뉴스는 "us-en")
            safesearch="off",
            timelimit="w",     # 최근 1주일 이내 뉴스만
            max_results=max_results
        )
        
        # 검색 결과 출력 (리스트 안의 딕셔너리 형태)
        for i, article in enumerate(results, 1):
            print(f"{i}. 제목: {article['title']}")
            print(f"   출처: {article['source']}")
            print(f"   날짜: {article['date']}")
            print(f"   본문 요약: {article['body']}\n")
            print("-" * 50)
            
        return list(results)

# 함수 실행! (원하는 검색어로 바꿔보세요)
news_data = get_recent_news("gold price", max_results=3)