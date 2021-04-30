# 차트 데이터 관리 모듈
class Environment:
    PRICE_IDX = 4  # 종가의 위치

    # __init__ 은 파이썬 컨스트럭터, 자동으로 호출됨, chart_data 2차원 배열, Pandas의 DataFrame임
    def __init__(self, chart_data=None):
        self.chart_data = chart_data # 주식 종목의 차트 데이터
        self.observation = None # 현재 관측치 (행)
        self.idx = -1 # 차트 데이터에서의 현재 위치 (컬럼)

    def reset(self):
        self.observation = None
        self.idx = -1

    def observe(self):
        # len 파이썬 내장함수, 문자열, 배열등의 길이 반환
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx] # iloc은 DataFrame함수로, 특정 행의 데이터를 가져옴.
            return self.observation
        return None

    # 종가 get 함수
    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None

    def set_chart_data(self, chart_data):
        self.chart_data = chart_data
