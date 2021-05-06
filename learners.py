# 파이선 기본 모듈
import os # 폴더 생성이나, 파일 경로 등을 위함
import logging # 학습 과정중에 정보를 기록하기 위함
import abc # 추상 클래스 정의용, ref: https://wikidocs.net/16075
import collections
import threading
import time # 학습 시간 측정용
import numpy as np # 배열 자료 구조를 위한, NumPy 라이브러리
from utils import sigmoid # 정책 신경망 학습 레이블 생성울 위함.
from environment import Environment
from agent import Agent
from networks import Network, DNN, LSTMNetwork, CNN
from visualizer import Visualizer

# 학습기 모듈
#
class ReinforcementLearner:
    __metaclass__ = abc.ABCMeta
    lock = threading.Lock()

    # rl_method: 강화학습 기법을 의미, 이 값은 하위 클래스에 따라 달라진다. (DQNLener는 dq, A2CLener는 ac 등)
    # stock_code: 학습을 진행하는 주식 종목 코드
    # chart_data: 주식 일봉 차트(환경에 해당)
    # training_data: 학습을 위해서 전처리된 데이터
    # min_trading_unit: 투자 최소 단위
    # max_trading_unit: 투자 최대 단위
    # delayed_reward_threshold: 지연 보상 임계값, 수익 or 손실률이 임계값보다 크면 지연 보상이 발생
    # mini_batch_size: ??
    # net: 신경망 종류, 이 값에 따라서, 가치 신경망, 정챙신경망으로 사용할 신경망 클래스가 달라짐
    # n_steps: LSTM, CNN 신경망에서 사용하는 샘플 묶음 크기
    # lr: (learn rate?), 학습 속도, 너무 크면 학습이 진행 안되고, 너무 작으면 학습이 오래 걸림
    # value_network, policy_network: 값을 들어오는 경우, 해당 모델을 가치 신경망, 정책신경망으로 사용
    # output_path: 학습 과정에서 발생하는 로그, 가시화 결과 및 학습 종료 후 저장되는 신겯망 모델 파일의 저장 위치 결정
    def __init__(self, rl_method='rl', stock_code=None,
                chart_data=None, training_data=None,
                min_trading_unit=1, max_trading_unit=2,
                delayed_reward_threshold=.05,
                net='dnn', num_steps=1, lr=0.001,
                value_network=None, policy_network=None,
                output_path='', reuse_models=True):
        # 인자 확인
        assert min_trading_unit > 0
        assert max_trading_unit > 0
        assert max_trading_unit >= min_trading_unit
        assert num_steps > 0
        assert lr > 0
        # 강화학습 기법 설정
        self.rl_method = rl_method
        # 환경 설정
        self.stock_code = stock_code # 강화학습 대상이 되는 주식 종목 코드
        self.chart_data = chart_data # 주식 종목의 차트 데이터
        self.environment = Environment(chart_data) # 강화학습 환경 객체
        # 에이전트 설정
        self.agent = Agent(self.environment,
                    min_trading_unit=min_trading_unit,
                    max_trading_unit=max_trading_unit,
                    delayed_reward_threshold=delayed_reward_threshold)
        # 학습 데이터
        self.training_data = training_data
        self.sample = None
        self.training_data_idx = -1
        # 벡터 크기 = 학습 데이터 벡터 크기 + 에이전트 상태 크기
        self.num_features = self.agent.STATE_DIM
        if self.training_data is not None:
            self.num_features += self.training_data.shape[1]
        # 신경망 클래스 객체는, 본 클래스의 하위 클래스에서 생성
        # 신경망 설정
        self.net = net
        self.num_steps = num_steps
        self.lr = lr
        self.value_network = value_network # 가치 신경망
        self.policy_network = policy_network # 정책 신경망
        self.reuse_models = reuse_models
        # 가시화 모듈
        self.visualizer = Visualizer()
        # 메모리
        # 강화 학습 과정에서 발생하는 각종 데이터를 쌓아두기 위해서, memory라는 변수 정의
        self.memory_sample = [] # 학습 데이터 샘플
        self.memory_action = [] # 수행한 행동
        self.memory_reward = [] # 획득한 보상
        self.memory_value = [] # 행동의 예측 가치
        self.memory_policy = [] # 핻동의 예측?확률?
        self.memory_pv = [] # 포트폴리오 가치
        self.memory_num_stocks = [] # 보유 주식수
        self.memory_exp_idx = [] # 탐험 위치
        self.memory_learning_idx = [] # 학습 위치
        # 에포크 관련 정보
        self.loss = 0. # 손실
        self.itr_cnt = 0 # 수익발생 횟수
        self.exploration_cnt = 0 # 탐험 횟수
        self.batch_size = 0 # 배치 크기?
        self.learning_cnt = 0 # 학습 횟수
        # 로그 등 출력 경로
        self.output_path = output_path

    # 가치 신경망 생성 함수
    # 팩토리 함수 느낌
    # 가치 신경망은, 손익율을 회귀분석하는 모델로 보면 된다. 따라서, activation은 선형, loss는 mse이다.???
    def init_value_network(self, shared_network=None,
            activation='linear', loss='mse'):
        if self.net == 'dnn':
            self.value_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.value_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.value_network = CNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and \
            os.path.exists(self.value_network_path): # reuse_models이 True이고, value_network_path 값이 있으면 신경망 모델 파일을 불러온다...
                self.value_network.load_model(
                    model_path=self.value_network_path)

    # 정책 신경망 생성 함수
    # activation이 sigmoid로 다르다.
    # 정책신경망은 PV을 높이기 위해 취하기 좋은 행동에 대한 '분류' 모델
    # 활성 함수로 sigmoid을 써서 0 ~ 1 시아의 값으로 확률로 사용하기 위함
    def init_policy_network(self, shared_network=None,
            activation='sigmoid', loss='binary_crossentropy'):
        if self.net == 'dnn':
            self.policy_network = DNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'lstm':
            self.policy_network = LSTMNetwork(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        elif self.net == 'cnn':
            self.policy_network = CNN(
                input_dim=self.num_features,
                output_dim=self.agent.NUM_ACTIONS,
                lr=self.lr, num_steps=self.num_steps,
                shared_network=shared_network,
                activation=activation, loss=loss)
        if self.reuse_models and \
            os.path.exists(self.policy_network_path):
            self.policy_network.load_model(
                model_path=self.policy_network_path)

    # 초기화 함수
    # 에포크 초기화 함수
    # 에포크마다 데이터가 새로 쌓이는 변수들을 초기화 한다.
    def reset(self):
        self.sample = None # 읽어온 학습 데이터가 샘플에 할당됨(초기화에선 None)
        self.training_data_idx = -1 # 학습 데이터를 처음부터 다시 읽기위해서 -1로 설정
        # 환경 초기화
        self.environment.reset() # 환경클래스의 reset호출
        # 에이전트 초기화
        self.agent.reset() # 에이전트가 제공하는 reset호출
        # 가시화 초기화
        self.visualizer.clear([0, len(self.chart_data)]) # 가시화 클래스의 clear호출
        # 메모리 초기화
        self.memory_sample = []
        self.memory_action = []
        self.memory_reward = []
        self.memory_value = []
        self.memory_policy = []
        self.memory_pv = []
        self.memory_num_stocks = []
        self.memory_exp_idx = []
        self.memory_learning_idx = []
        # 에포크 관련 정보 초기화
        self.loss = 0. # 신경망의 결과가 학습 데이터와 얼마나 차이가 있는지를 저장하는 변수 loss가 줄어야 좋은거임!
        self.itr_cnt = 0 # 수행한 에포크 수를 저장
        self.exploration_cnt = 0 # 탐험 수 저장, epsilon이 0.1dlrh 100번 투자 결정이 있다고 한다면 약 10번의 무작위 투자
        self.batch_size = 0 # 학습할 미니 배치 크기
        self.learning_cnt = 0 # 한 에포크 동안 수행한 미니 배치 학습 횟수

    # 환경 객체에서 샘플을 획득하는 함수
    # 학습 데이터플 구성하는 샘플 하나를 생성하는 함수
    def build_sample(self):
        self.environment.observe() # 차트 데이터의 현재 인덱스에서, 다음 인덱스 데이터를 읽게한다.
        if len(self.training_data) > self.training_data_idx + 1: # 학습 데이터 존재 확인
            self.training_data_idx += 1
            self.sample = self.training_data.iloc[self.training_data_idx].tolist() # 샘플 가져옴, 샘플은 26개의 값임
            self.sample.extend(self.agent.get_states()) # 에이전트에서 2개 값을 추가! (28개값!)
            return self.sample
        return None

    # 배치 학습 데이터 생성 함수, 추상 메소드로 하위 클래스가 반드시 구현해야 한다.!
    @abc.abstractmethod
    def get_batch(self, batch_size, delayed_reward, discount_factor):
        pass

    # 가치 신경망 및 정책 신경망 학습 함수
    # get_batch을 호출해서 배치 학습 데이터를 생성
    # 가치 신경망 및 정책 신경망의 train_on_batch을 호출하여, 학습 시킴
    # 가치 신경망: DQN, AC, A2C
    # 정책 신경망: PolicyGradient, AC, A2C
    def update_networks(self,
            batch_size, delayed_reward, discount_factor):
        # 배치 학습 데이터 생성
        x, y_value, y_policy = self.get_batch(
            batch_size, delayed_reward, discount_factor)
        if len(x) > 0:
            loss = 0
            if y_value is not None:
                # 가치 신경망 갱신
                loss += self.value_network.train_on_batch(x, y_value)
            if y_policy is not None:
                # 정책 신경망 갱신
                loss += self.policy_network.train_on_batch(x, y_policy)
            return loss # 학습 후 손실 반환
        return None

    # 가치 신경망 및 정책 신경망 학습 요청 함수
    # 배치 학습 데이터의 크기를 정하고, update_networks 호출(위함수)
    # _loss에 로 총 loss을 생성
    def fit(self, delayed_reward, discount_factor, full=False):
        batch_size = len(self.memory_reward) if full \
            else self.batch_size
        # 배치 학습 데이터 생성 및 신경망 갱신
        if batch_size > 0:
            _loss = self.update_networks(
                batch_size, delayed_reward, discount_factor)
            if _loss is not None:
                self.loss += abs(_loss)
                self.learning_cnt += 1 # 학습 횟수 저장, loss / learning_cnt하면 에포크의 학습 손실을 알 수 있음
                self.memory_learning_idx.append(self.training_data_idx) # 학습 위치 저장
            self.batch_size = 0

    # 에포크 정보 가시화 함수
    def visualize(self, epoch_str, num_epoches, epsilon):
        self.memory_action = [Agent.ACTION_HOLD] \
            * (self.num_steps - 1) + self.memory_action
        self.memory_num_stocks = [0] * (self.num_steps - 1) \
            + self.memory_num_stocks
        if self.value_network is not None:
            self.memory_value = [np.array([np.nan] \
                * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                    + self.memory_value
        if self.policy_network is not None:
            self.memory_policy = [np.array([np.nan] \
                * len(Agent.ACTIONS))] * (self.num_steps - 1) \
                    + self.memory_policy
        self.memory_pv = [self.agent.initial_balance] \
            * (self.num_steps - 1) + self.memory_pv
        self.visualizer.plot(
            epoch_str=epoch_str, num_epoches=num_epoches,
            epsilon=epsilon, action_list=Agent.ACTIONS,
            actions=self.memory_action,
            num_stocks=self.memory_num_stocks,
            outvals_value=self.memory_value,
            outvals_policy=self.memory_policy,
            exps=self.memory_exp_idx,
            learning_idxes=self.memory_learning_idx,
            initial_balance=self.agent.initial_balance,
            pvs=self.memory_pv,
        )
        self.visualizer.save(os.path.join(
            self.epoch_summary_dir,
            'epoch_summary_{}.png'.format(epoch_str))
        )

    # 강화학습 수행 함수
    # 핵심 함수!
    def run(
        self,
        num_epoches=100, # 총 수행할 반복 학습 횟수, 너무 크면 학습에 걸리는 시간이 길어짐
        balance=10000000, # 초기 투자금
        discount_factor=0.9, # 상태-행동 가치를 구할때 적용할 할인율, 과거로 갈수록 현재 보상을 약하게 적용한다.
        start_epsilon=0.5, # 초기 탐험 비율
        learning=True # 학습을 마치면 학습된 가치 신경망모델, 정책 신경망 모델이 생성된다. 이런 신경망 모델으 만들꺼면 True, 이미 학습된 모델로, 투자 시뮬레이션일때는 False
    ):
        info = "[{code}] RL:{rl} Net:{net} LR:{lr} " \
            "DF:{discount_factor} TU:[{min_trading_unit}," \
            "{max_trading_unit}] DRT:{delayed_reward_threshold}".format(
            code=self.stock_code, rl=self.rl_method, net=self.net,
            lr=self.lr, discount_factor=discount_factor,
            min_trading_unit=self.agent.min_trading_unit,
            max_trading_unit=self.agent.max_trading_unit,
            delayed_reward_threshold=self.agent.delayed_reward_threshold
        )
        with self.lock:
            logging.info(info) # 강화 학습의 설정값을 로깅 한다.

        # 시작 시간
        time_start = time.time()

        # 가시화 준비
        # 차트 데이터는 변하지 않으므로 미리 가시화
        self.visualizer.prepare(self.environment.chart_data, info)

        # 가시화 결과 저장할 폴더 준비
        # epoch_summary_ 라는 폴더에 저장
        self.epoch_summary_dir = os.path.join(
            self.output_path, 'epoch_summary_{}'.format(
                self.stock_code))
        if not os.path.isdir(self.epoch_summary_dir):
            os.makedirs(self.epoch_summary_dir)
        else:
            for f in os.listdir(self.epoch_summary_dir):
                os.remove(os.path.join(self.epoch_summary_dir, f))

        # 에이전트 초기 자본금 설정
        self.agent.set_balance(balance)

        # 학습에 대한 정보 초기화
        max_portfolio_value = 0
        epoch_win_cnt = 0

        # 학습 반복
        for epoch in range(num_epoches):
            time_start_epoch = time.time()

            # step 샘플을 만들기 위한 큐
            # deque사용 - 참고: https://opensourcedev.tistory.com/3
            q_sample = collections.deque(maxlen=self.num_steps)

            # 환경, 에이전트, 신경망, 가시화, 메모리 초기화
            self.reset()

            # 학습을 진행할 수록 탐험 비율 감소
            if learning:
                epsilon = start_epsilon \
                    * (1. - float(epoch) / (num_epoches - 1))
                self.agent.reset_exploration()
            else:
                epsilon = start_epsilon
                self.agent.reset_exploration(alpha=0)

            while True:
                # 샘플 생성
                next_sample = self.build_sample()
                if next_sample is None:
                    break # 샘플 만큼 while문 반복

                # num_steps만큼 샘플 저장
                q_sample.append(next_sample)
                if len(q_sample) < self.num_steps:
                    continue

                # 가치, 정책 신경망 예측
                # 각 신경망의 predict함수 호출
                pred_value = None
                pred_policy = None
                if self.value_network is not None:
                    pred_value = self.value_network.predict(
                        list(q_sample))
                if self.policy_network is not None:
                    pred_policy = self.policy_network.predict(
                        list(q_sample))

                # 신경망 또는 탐험에 의한 행동 결정
                # 행동, 결정에 대한 확신도, 무작위 탐험 유무
                action, confidence, exploration = \
                    self.agent.decide_action(
                        pred_value, pred_policy, epsilon)

                # 결정한 행동을 수행하고 즉시 보상과 지연 보상 획득
                immediate_reward, delayed_reward = \
                    self.agent.act(action, confidence)

                # 행동 및 행동에 대한 결과를 기억
                self.memory_sample.append(list(q_sample))
                self.memory_action.append(action)
                self.memory_reward.append(immediate_reward)
                if self.value_network is not None:
                    self.memory_value.append(pred_value)
                if self.policy_network is not None:
                    self.memory_policy.append(pred_policy)
                self.memory_pv.append(self.agent.portfolio_value)
                self.memory_num_stocks.append(self.agent.num_stocks)
                if exploration:
                    self.memory_exp_idx.append(self.training_data_idx)

                # 반복에 대한 정보 갱신
                self.batch_size += 1
                self.itr_cnt += 1
                self.exploration_cnt += 1 if exploration else 0 # 3항연산???

                # 지연 보상 발생된 경우 미니 배치 학습
                # 지연보상은 지연보상 임계치가 넘는 손익률이 발생하면 주어진다.
                if learning and (delayed_reward != 0):
                    self.fit(delayed_reward, discount_factor)

            # 에포크 종료 후 학습 (while문 이후 미니 배치 학습)
            if learning:
                self.fit(
                    self.agent.profitloss, discount_factor, full=True)

            # 에포크 관련 정보 로그 기록
            num_epoches_digit = len(str(num_epoches))
            epoch_str = str(epoch + 1).rjust(num_epoches_digit, '0') # 문자열을 자리수에 맞게 정렬(우측) 함수
            time_end_epoch = time.time()
            elapsed_time_epoch = time_end_epoch - time_start_epoch
            if self.learning_cnt > 0:
                self.loss /= self.learning_cnt
            logging.info("[{}][Epoch {}/{}] Epsilon:{:.4f} "
                "#Expl.:{}/{} #Buy:{} #Sell:{} #Hold:{} "
                "#Stocks:{} PV:{:,.0f} "
                "LC:{} Loss:{:.6f} ET:{:.4f}".format(
                    self.stock_code, epoch_str, num_epoches, epsilon,
                    self.exploration_cnt, self.itr_cnt,
                    self.agent.num_buy, self.agent.num_sell,
                    self.agent.num_hold, self.agent.num_stocks,
                    self.agent.portfolio_value, self.learning_cnt,
                    self.loss, elapsed_time_epoch))

            # 에포크 관련 정보 가시화
            self.visualize(epoch_str, num_epoches, epsilon)

            # 학습 관련 정보 갱신
            max_portfolio_value = max(
                max_portfolio_value, self.agent.portfolio_value)
            if self.agent.portfolio_value > self.agent.initial_balance:
                epoch_win_cnt += 1

        # 종료 시간
        time_end = time.time()
        elapsed_time = time_end - time_start

        # 학습 관련 정보 로그 기록
        with self.lock:
            logging.info("[{code}] Elapsed Time:{elapsed_time:.4f} "
                "Max PV:{max_pv:,.0f} #Win:{cnt_win}".format(
                code=self.stock_code, elapsed_time=elapsed_time,
                max_pv=max_portfolio_value, cnt_win=epoch_win_cnt))

    # 가치 신경망 및 정책 신경망 저장 함수
    def save_models(self):
        if self.value_network is not None and \
                self.value_network_path is not None:
            self.value_network.save_model(self.value_network_path)
        if self.policy_network is not None and \
                self.policy_network_path is not None:
            self.policy_network.save_model(self.policy_network_path)


# 가치 신경망으로만 학습
class DQNLearner(ReinforcementLearner):
    def __init__(self, *args, value_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_network_path = value_network_path
        self.init_value_network()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]), # 배열 요소를 역순으로 바꾸고 zip으로 묵음 (근데 왜 역순으로 바꾸는지???)
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features)) # 0 세팅?, (num_features => 학습 데이터 28차원?), num_steps: 샘플 묶음 크기?
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS)) # 0 세팅? (행동 수?)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, reward) in enumerate(memory): # mem에서 sample, acrion, value, reward꺼내옴, 근데 이때 꺼내온건 역순!(학습 데이터의 마지막 부분)
            # get_batch는 학습 데이터 생성용 함수
            # memory_ 들은 하나의 에포크에서, 값들을 기록, 배열의 맨뒤가 가장 마지막 학습 run임(한 에포크 내에서...)
            # 이게 run의 에포크 for문에서 while전에 하나의 에포크마다 reset됨(memory는...)
            # get_batch는 update_network함수 내에서 fit(미니배치 학습)에서 불림
            # 따라서 하나의 에포크 while문에서 지연보상 or 에포크 하나 끝나고, 미니배치 학습 할때 호출
            # 따라서, 하나의 에포크 내에서 설정한 마지막 mem값(while라스트) 부터 꺼내온다는 의미
            x[i] = sample
            y_value[i] = value
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, None


# 정책 경사 강화 학습 클래스
class PolicyGradientLearner(ReinforcementLearner):
    def __init__(self, *args, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_network_path = policy_network_path
        self.init_policy_network()

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        # batch_size: 지연보상이 발생할때 결정된다. 따라서 매번 다르다.
        # 아.. 결국 reversed가 batch_size만큼만 사용? 하니깐 지연 보상 용까지만 잘라서 그거로 미니학습? 데이터에 사용?
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features)) # 학습 데이터(feature 는 28차원) 고정
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5) # action도 고정...
        reward_next = self.memory_reward[-1]
        for i, (sample, action, policy, reward) in enumerate(memory):
            x[i] = sample
            y_policy[i] = policy
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_policy[i, action] = sigmoid(r)
            reward_next = reward
        return x, None, y_policy  # 가치신경망은 없어서 두번째꺼 None임.


class ActorCriticLearner(ReinforcementLearner):
    def __init__(self, *args, shared_network=None,
        value_network_path=None, policy_network_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if shared_network is None:
            self.shared_network = Network.get_shared_network(
                net=self.net, num_steps=self.num_steps,
                input_dim=self.num_features)
        else:
            self.shared_network = shared_network
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=shared_network)

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) \
            in enumerate(memory):
            x[i] = sample
            y_value[i] = value
            y_policy[i] = policy
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            y_policy[i, action] = sigmoid(value[action])
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy


class A2CLearner(ActorCriticLearner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_batch(self, batch_size, delayed_reward, discount_factor):
        memory = zip(
            reversed(self.memory_sample[-batch_size:]),
            reversed(self.memory_action[-batch_size:]),
            reversed(self.memory_value[-batch_size:]),
            reversed(self.memory_policy[-batch_size:]),
            reversed(self.memory_reward[-batch_size:]),
        )
        x = np.zeros((batch_size, self.num_steps, self.num_features))
        y_value = np.zeros((batch_size, self.agent.NUM_ACTIONS))
        y_policy = np.full((batch_size, self.agent.NUM_ACTIONS), .5)
        value_max_next = 0
        reward_next = self.memory_reward[-1]
        for i, (sample, action, value, policy, reward) \
            in enumerate(memory):
            x[i] = sample
            r = (delayed_reward + reward_next - reward * 2) * 100
            y_value[i, action] = r + discount_factor * value_max_next
            advantage = value[action] - value.mean()
            y_policy[i, action] = sigmoid(advantage)
            value_max_next = value.max()
            reward_next = reward
        return x, y_value, y_policy


class A3CLearner(ReinforcementLearner):
    def __init__(self, *args, list_stock_code=None,
        list_chart_data=None, list_training_data=None,
        list_min_trading_unit=None, list_max_trading_unit=None,
        value_network_path=None, policy_network_path=None,
        **kwargs):
        assert len(list_training_data) > 0
        super().__init__(*args, **kwargs)
        self.num_features += list_training_data[0].shape[1]

        # 공유 신경망 생성
        self.shared_network = Network.get_shared_network(
            net=self.net, num_steps=self.num_steps,
            input_dim=self.num_features)
        self.value_network_path = value_network_path
        self.policy_network_path = policy_network_path
        if self.value_network is None:
            self.init_value_network(shared_network=self.shared_network)
        if self.policy_network is None:
            self.init_policy_network(shared_network=self.shared_network)

        # A2CLearner 생성
        self.learners = []
        for (stock_code, chart_data, training_data,
            min_trading_unit, max_trading_unit) in zip(
                list_stock_code, list_chart_data, list_training_data,
                list_min_trading_unit, list_max_trading_unit
            ):
            learner = A2CLearner(*args,
                stock_code=stock_code, chart_data=chart_data,
                training_data=training_data,
                min_trading_unit=min_trading_unit,
                max_trading_unit=max_trading_unit,
                shared_network=self.shared_network,
                value_network=self.value_network,
                policy_network=self.policy_network, **kwargs)
            self.learners.append(learner)

    def run(
        self, num_epoches=100, balance=10000000,
        discount_factor=0.9, start_epsilon=0.5, learning=True):
        threads = []
        for learner in self.learners:
            threads.append(threading.Thread(
                target=learner.run, daemon=True, kwargs={
                'num_epoches': num_epoches, 'balance': balance,
                'discount_factor': discount_factor,
                'start_epsilon': start_epsilon,
                'learning': learning
            }))
        for thread in threads:
            thread.start()
            time.sleep(1)
        for thread in threads: thread.join()
