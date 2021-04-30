import numpy as np
import utils

# 투자 행동을 수행하고, 투자금, 보유주식을 관리
class Agent:
    # 에이전트 상태가 구성하는 값 개수
    STATE_DIM = 2  # 주식 보유 비율, 포트폴리오 가치 비율, 2개를 가져서 2(2차원)

    # 매매 수수료 및 세금
    TRADING_CHARGE = 0.00015  # 거래 수수료 0.015%
    # TRADING_CHARGE = 0.00011  # 거래 수수료 0.011%
    # TRADING_CHARGE = 0  # 거래 수수료 미적용
    TRADING_TAX = 0.0025  # 거래세 0.25%
    # TRADING_TAX = 0  # 거래세 미적용

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 홀딩
    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL]
    NUM_ACTIONS = len(ACTIONS)  # 인공 신경망에서 고려할 출력값의 개수 (매도, 매수) 2개임 (일단...)

    # 생성자
    def __init__(
        self, environment, min_trading_unit=1, max_trading_unit=2,
        delayed_reward_threshold=.05):
        # Environment 객체
        # 현재 주식 가격을 가져오기 위해 환경 참조
        self.environment = environment

        # 최소 매매 단위, 최대 매매 단위, 지연보상 임계치
        self.min_trading_unit = min_trading_unit  # 최소 단일 거래 단위
        self.max_trading_unit = max_trading_unit  # 최대 단일 거래 단위
        # 지연보상 임계치, 손익율이 이 값을 넘어가면 지연 보상을 발생
        self.delayed_reward_threshold = delayed_reward_threshold

        # Agent 클래스의 속성
        self.initial_balance = 0  # 초기 자본금
        self.balance = 0  # 현재 현금 잔고
        self.num_stocks = 0  # 보유 주식 수
        # 포트폴리오 가치: balance + num_stocks * {현재 주식 가격}
        self.portfolio_value = 0
        self.base_portfolio_value = 0  # 직전 학습 시점의 PV, 과거 Pv로 현내 Pv의 증가, 감소 비교의 기준
        self.num_buy = 0  # 매수 횟수
        self.num_sell = 0  # 매도 횟수
        self.num_hold = 0  # 홀딩 횟수
        self.immediate_reward = 0  # 즉시 보상, 에이전트가 가장 최근 행한 행동에 대한 즉시 보상값
        self.profitloss = 0  # 현재 손익
        self.base_profitloss = 0  # 직전 지연 보상 이후 손익
        self.exploration_base = 0  # 탐험 행동 결정 기준, 탐험을 하더라도, 매수를 기조로 할지, 매도를 기조로 할지 정하는 부분

        # Agent 클래스의 상태
        self.ratio_hold = 0  # 주식 보유 비율
        self.ratio_portfolio_value = 0  # 포트폴리오 가치 비율

    # 에이전트의 속성 초기화, 학습 단계에서 한 에포크 마다 상태를 초기화 해주어야 한다.
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.base_portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.immediate_reward = 0
        self.ratio_hold = 0
        self.ratio_portfolio_value = 0

    # 탐험의 기준이 되는 exploration_base을 새로 정하는 함수
    def reset_exploration(self, alpha=None):
        if alpha is None:
            alpha = np.random.rand() / 2

        # 0.5는 매수 기준 50% 매수 탑험 확률을 하드코딩 부여한것
        self.exploration_base = 0.5 + alpha

    # 초기 자본금 설정
    def set_balance(self, balance):
        self.initial_balance = balance

    # 에이전트의 상태 반환
    def get_states(self):
        # int 내장함수, 정수 리턴 (0.5 => 0), (1.3 => 1)
        self.ratio_hold = self.num_stocks / int(
            self.portfolio_value / self.environment.get_price())
        self.ratio_portfolio_value = (
            self.portfolio_value / self.base_portfolio_value
        )
        # ratio_hold
        # 주식 보유 비율, 0이면 주식 미보유, 0.5면 최대로 가질수 있는 주식 대비 절반 보유, 1이면 최대 보유.
        # 보유 비율이 적으면, 매수 관점의 투자, 보유 비율이 크면 매도 관점의 투자 -> 정책신경망의 입력으로 사용됨

        # ratio_portfolio_value
        # 포트폴리오 가치 비율 = 포트폴리오 가치 / 기준 포트폴리오 가치
        # 기준 포트폴리오 가치: 직전에 목표 수익 또는 손익을 달성했을때의 가치 -> 현재 수익 or 손실의 판단 기준
        # 포트폴리오 가치 비율이 0에 가까우면: 손실 || 1에 가까우면 수익
        # 수익율이 목표 수익율에 가까우면 매도의 관점에서 투자
        # 수익율이 행동 결정에 영향을 줄 수 있음 -> 정책신경망의 입력으로 사용

        # 파이썬 (a, b, c, ...) 은 튜플(tuple), 리스트[]와 비슷한데, 추가, 변경, 삭제 불가
        return (
            self.ratio_hold,
            self.ratio_portfolio_value
        )

    # 에이전트가 행동을 결정하고, 유효성을 검사하는 함수
    # 입력으로 들어온 epsilon의 확률로, 무작위 행동을 결정하고, 그렇지 않은 경우 신경망을 통해 행동을 결정
    def decide_action(self, pred_value, pred_policy, epsilon):
        # confidence: 결정한 행동의 신뢰 값?
        confidence = 0.

        # pred_policy: 정책 신경망의 출력
        # pred_value: ?? 정책 신경망 값?
        # 정책신경망의 출력이 있으면 pred_policy로 행동 결정, 없으면 pred_value로 행동결정
        # DQNLearner 은 pred_policy가 None이라 pred_value로 결정
        pred = pred_policy
        if pred is None:
            pred = pred_value

        if pred is None:
            # 예측 값이 없을 경우 탐험
            epsilon = 1
        else:
            # 값이 모두 같은 경우 탐험
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # 탐험 결정
        # NumPy의 rand 함수: 0 ~ 1 사이의 값 랜덤 생성 반환
        # NumPy의 randint(low, high=None): low만 있으면, 0 ~ low까지 랜덤 수 생성, low, high다 있으면 low ~ high까지의 랜덤 수 생성
        # argmax: 입력 배열의 가장 큰 값의 위치 반환: e.g) np.argmax([3, 5, 7, 0 , -3])면 2가 리턴됨
        if np.random.rand() < epsilon:
            exploration = True
            # exploration_base 기준으로 Buy액션임으로, exploration_base가 1에 가까울수록, 매수 행동을 더 많이 하게 될꺼임.
            # 반대로 exploration_base가 0에 가까울수록, 매도 행동을 더 많이 할꺼임.
            if np.random.rand() < self.exploration_base:
                action = self.ACTION_BUY
            else:
                action = np.random.randint(self.NUM_ACTIONS - 1) + 1
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration # 괄호()가 없어도 튜플...?

    # 결정한 행동의 유효성 검사 함수, 신용이나 공매도는 미고려
    # 행동(매수, 매도) 가 결정되어도, 실제 해동을 수행 못할 수 있다 이를 검사하는 함수
    def validate_action(self, action):
        if action == Agent.ACTION_BUY:
            # 적어도 1주를 살 수 있는지 확인
            # 거래 수수료도 같이 고려한다
            if self.balance < self.environment.get_price() * (
                1 + self.TRADING_CHARGE) * self.min_trading_unit:
                return False
        elif action == Agent.ACTION_SELL:
            # 주식 잔고가 있는지 확인
            if self.num_stocks <= 0:
                return False
        return True

    # 정책 신경망이 결정한 행동의 신뢰가 높을수록 매수 or 매도 단위를 크게 변경한다.
    def decide_trading_unit(self, confidence):
        # 신뢰값이 없으면, 매수/매도가 최소단위로
        # 신뢰값이 있으면, 최대, 최소 값의 차를 신뢰도에 따라서, 차이만큼 혹은 신뢰도 만큼 더 더해서 단위 변경
        if np.isnan(confidence):
            return self.min_trading_unit
        added_traiding = max(min(
            int(confidence * (self.max_trading_unit -
                self.min_trading_unit)),
            self.max_trading_unit-self.min_trading_unit
        ), 0)
        return self.min_trading_unit + added_traiding

    # 투자 행동을 하는 함수
    # action은 매도(0) 또는 매수(1)
    # confidence: 정책신경망으로 결정된 행동이면, 결정한 행동에 대한 소프트맥스 확률값???
    def act(self, action, confidence):
        # 행동 가능한지 확인, 불가능시 행동을 관망으로 변경
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # 환경에서 현재 가격 얻기 (현재 주가!!!)
        curr_price = self.environment.get_price()

        # 즉시 보상 초기화
        self.immediate_reward = 0

        # 매수
        if action == Agent.ACTION_BUY:
            # 매수할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price * (1 + self.TRADING_CHARGE) \
                    * trading_unit
            )
            # 보유 현금이 모자랄 경우 보유 현금으로 가능한 만큼 최대한 매수
            if balance < 0:
                trading_unit = max(
                    min(
                        int(self.balance / (
                            curr_price * (1 + self.TRADING_CHARGE))),
                        self.max_trading_unit
                    ),
                    self.min_trading_unit
                )
            # 수수료를 적용하여 총 매수 금액 산정
            invest_amount = curr_price * (1 + self.TRADING_CHARGE) \
                * trading_unit
            if invest_amount > 0:
                self.balance -= invest_amount  # 보유 현금을 갱신
                self.num_stocks += trading_unit  # 보유 주식 수를 갱신
                self.num_buy += 1  # 매수 횟수 증가

        # 매도
        elif action == Agent.ACTION_SELL:
            # 매도할 단위를 판단
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 모자랄 경우 가능한 만큼 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)
            # 매도
            invest_amount = curr_price * (
                1 - (self.TRADING_TAX + self.TRADING_CHARGE)) \
                    * trading_unit
            if invest_amount > 0:
                self.num_stocks -= trading_unit  # 보유 주식 수를 갱신
                self.balance += invest_amount  # 보유 현금을 갱신
                self.num_sell += 1  # 매도 횟수 증가

        # 홀딩
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 홀딩 횟수 증가

        # 포트폴리오 가치 갱신, 매수, 매도, 홀딩 모두 포트폴리오 가치는 변화 될 수 있음
        # profitloss 포트폴리오 가치 등락률?
        self.portfolio_value = self.balance + curr_price \
            * self.num_stocks
        self.profitloss = (
            (self.portfolio_value - self.initial_balance) \
                / self.initial_balance
        )

        # 즉시 보상 - 수익률, (기준 포트폴리오 가치 대비 현재 포트폴리오의 가치 비율 (등락율?))
        self.immediate_reward = self.profitloss

        # 지연 보상 - 익절, 손절 기준
        # 지연 보상은, 즉시 보상이 지연보상 임계치인 delayed_reward_threshold을 초과하면 즉시 보상 그외 0
        delayed_reward = 0
        self.base_profitloss = (
            (self.portfolio_value - self.base_portfolio_value) \
                / self.base_portfolio_value
        )
        if self.base_profitloss > self.delayed_reward_threshold or \
            self.base_profitloss < -self.delayed_reward_threshold:
            # 목표 수익률 달성하여 기준 포트폴리오 가치 갱신
            # 또는 손실 기준치를 초과하여 기준 포트폴리오 가치 갱신
            self.base_portfolio_value = self.portfolio_value
            delayed_reward = self.immediate_reward
        else:
            delayed_reward = 0

        return self.immediate_reward, delayed_reward
