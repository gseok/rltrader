import os
import threading
import numpy as np


class DummyGraph:
    def as_default(self): return self
    def __enter__(self): pass
    def __exit__(self, type, value, traceback): pass

def set_session(sess): pass


graph = DummyGraph()
sess = None

# 케라스 import하는 부분
# 케라스는 텐서플로의 wrapper같은놈임!
# Dense, LSTM, Conv2D 는 신경망 요소
# 케라스 백엔드 이름 보고 tensorflow나 plaidml을 로드
# plaidml을 로드하는 경우엔 그래프가 없어서 더버그래프 생성한거임(위에 코드)
# graph?
#  tensorflow에서 신경망 모델을 정의하기 위한 공간을 graphㅏ고 한다.
# session?
#  정의한 모델을 시행하는 공간
if os.environ['KERAS_BACKEND'] == 'tensorflow':
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.backend import set_session
    import tensorflow as tf
    graph = tf.get_default_graph()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
elif os.environ['KERAS_BACKEND'] == 'plaidml.keras.backend':
    from keras.models import Model
    from keras.layers import Input, Dense, LSTM, Conv2D, \
        BatchNormalization, Dropout, MaxPooling2D, Flatten
    from keras.optimizers import SGD


# 신경망 클래스의 공통(부모) 클래스
class Network:
    # A3C에서는 스레드를 이용하여 병렬로 신경망을 쓰기 때문에, 스레드간 충돌 방지를 위해서 lock 클래스 객체를 가진다.
    lock = threading.Lock()

    def __init__(self, input_dim=0, output_dim=0, lr=0.001,
                shared_network=None, activation='sigmoid', loss='mse'):
        self.input_dim = input_dim # 입력 데이터 크기
        self.output_dim = output_dim # 출력 데이터 크기
        self.shared_network = shared_network # 신경망의 상단부, 여러 신경망이 공유 가능, e.g) a2c 가치신경망, 정책신경망이 신경망 상단부를 공유하고, 하단 부분만 가치 예측, 확률 예측을 위해 달라짐
        self.activation = activation # 신경망 출력 layer함수 이름('linear', 'sigmoid' 등..)
        self.loss = loss # 신경망 손실 함수
        self.lr = lr # 신경망 학습 속도
        self.model = None # Keras 라이브러리로 구성한 최종 신경망 모델

    # 신경망을 통해서 투자 행동별 가치나 확률 계산
    def predict(self, sample):
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                return self.model.predict(sample).flatten()

    # 배치 학습을 위한 데이터 생성
    # 학습 데이터, 레이블 x, y를 입력으로 받는다
    def train_on_batch(self, x, y):
        loss = 0.
        # with 구문(파이선): 참고: https://wikidocs.net/16078
        # with 구문은 주어진 객체의 __enter__() 을 호출해주고, with구문 빠져나올때 __exit__() 을 호출해준다.
        # 아래코드는 lock객체의 __enter__()가 호출되고, with최하단에서 with빠져나올때, lock객체의 __exit__()을 호출한다.
        # with <Lock 객체>: Thread safe code 가 된다.
        with self.lock:
            with graph.as_default():
                if sess is not None:
                    set_session(sess)
                loss = self.model.train_on_batch(x, y) # 케라스 모델(Model) 클래스 객체 내의 train_on_batch호출이고, 이는 입력 학습 데이터 집합을을 배치로 신경망 학습을 한다.
        return loss

    # 학습한 신경망을 파일로 저장
    def save_model(self, model_path):
        if model_path is not None and self.model is not None:
            self.model.save_weights(model_path, overwrite=True) # 케이스 save_weights 함수, 인공지능 신경망 모델을 HDF5 파일 형태로 저장한다

    # 파일로 저장한 신경망을 로드
    def load_model(self, model_path):
        if model_path is not None:
            self.model.load_weights(model_path)

    # 신경망의 상단부를 생성하는 클래스 메서드
    # 클래스 메서드는 정적 메서드처럼 인스턴스 없이 호출할 수 있다.
    # 클래스 메서드는 메서드 안에서 클래스 속성, 클래스 메서드에 접근해야 할 때 사용합니다.
    # 특히 cls를 사용하면 메서드 안에서 현재 클래스의 인스턴스를 만들 수도 있습니다. 즉, cls는 클래스이므로 cls()는 Person()과 같습니다.
    # 클래스 메서드에서는, self.aaaa 는 사용 불가 (인스턴스 없어서 안됨...@)
    # 공유 신경망을 리턴하는 함수, 각 신겨망의 공유 신경망을 생성(리턴) 한다.
    @classmethod
    def get_shared_network(cls, net='dnn', num_steps=1, input_dim=0):
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            if net == 'dnn':
                return DNN.get_network_head(Input((input_dim,)))
            elif net == 'lstm':
                return LSTMNetwork.get_network_head(
                    Input((num_steps, input_dim)))
            elif net == 'cnn':
                return CNN.get_network_head(
                    Input((1, num_steps, input_dim)))


# Network을 상속한 DNN 클래스
class DNN(Network):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.input_dim,))
                output = self.get_network_head(inp).output # 공유 신경망이 없으면 get_network_head로 직접 생성
            else:
                inp = self.shared_network.input # 공유 신경망이 있으면 해당 신경망의 input, output사용
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    # 신경망 구축
    # 파이썬 - https://wikidocs.net/16074
    # 레이어 차원, 활성화 함수, 드롭아웃 비율은 얼마든지 수정 가능하다.
    # 케라스 Model: 전체 신경망을 구성하는 클래스
    # Dense: 하나의 Node을 Dense로 구성, (신경망 구성상 하나의 원(하나의 퍼셉트론))
    @staticmethod
    def get_network_head(inp):
        output = Dense(256, activation='sigmoid', # Dense 레이어, 레이어 함수는 sigmoid
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output) # 배치 정규화로 학습 안정화 하기
        output = Dropout(0.1)(output) # Dropout 은 과적합을 일정 부분 피하게 하는것
        output = Dense(128, activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(64, activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        output = Dense(32, activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = Dropout(0.1)(output)
        return Model(inp, output)

    # 배치 학습 함수
    # 학습 데이터나, 샘플 형태를 적절히 변경(reshape)하고, 상위클래스의 함수를 그대로 호출한다.
    # DNN은 배치크기, 자질 벡터 차원의 모습을 가진다.
    # np.array: 파이선 기본 [] 을 NumPy형 []로 바꾸어줌
    #  e.g) np.array([1,2,3,4,5,6,]).reshape(3,2) => [[1,2], [3,4], [5,6]]
    #  주의-> 총량 item 6, 은 바꾸면 안됨..!!
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.input_dim))
        return super().train_on_batch(x, y)

    # 예측? 함수
    # 신경망을 통해서 투자 행동별 가치나 확률 계산
    def predict(self, sample):
        sample = np.array(sample).reshape((1, self.input_dim))
        return super().predict(sample)


class LSTMNetwork(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            # DNN과 다른 부분, 몇개의 샘플을 묶어서 LSTM 신경망 입력으로 사용할지 결정 한다.
            # train_on_batch, predict 함수에서 학습 데이터와 샘플 형태를 변경할때, num_steps사용
            self.num_steps = num_steps
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.num_steps, self.input_dim))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    # LSTM으로 구성
    # 여러개의 LSTM구성시, 맨마지막 LSTM을 제외하고 모두 return_sequences=True 주어야함
    # return_sequences=True을 주면 레이어 출력 개수를 num_steps만큼 유지한다.
    @staticmethod
    def get_network_head(inp):
        output = LSTM(256, dropout=0.1,
            return_sequences=True, stateful=False,
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = LSTM(128, dropout=0.1,
            return_sequences=True, stateful=False,
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(64, dropout=0.1,
            return_sequences=True, stateful=False,
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = LSTM(32, dropout=0.1,
            stateful=False,
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        return Model(inp, output)

    # LSTM은 (배치크기, 스텝수, 자질 백터 차원) 형태로 학습 데이터를 가진다.
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (1, self.num_steps, self.input_dim))
        return super().predict(sample)


# LSTM 처럼 다차원 데이터 처리 가능
# 2차원 데이터 사용하도록 Conv2D 사용
class CNN(Network):
    def __init__(self, *args, num_steps=1, **kwargs):
        super().__init__(*args, **kwargs)
        with graph.as_default():
            if sess is not None:
                set_session(sess)
            self.num_steps = num_steps # 2차원 크기 조절용으로 사용
            inp = None
            output = None
            if self.shared_network is None:
                inp = Input((self.num_steps, self.input_dim, 1))
                output = self.get_network_head(inp).output
            else:
                inp = self.shared_network.input
                output = self.shared_network.output
            output = Dense(
                self.output_dim, activation=self.activation,
                kernel_initializer='random_normal')(output)
            self.model = Model(inp, output)
            self.model.compile(
                optimizer=SGD(lr=self.lr), loss=self.loss)

    # 2차원 합성곱 레이어를 여러 겹 쌓아올린 구조
    # Conv2D의 padding='same'로 입력-출력 크기를 같게 설정
    # 합성곱 윈도우 사이즈는 kernel_size=(1, 5)
    # 모든 신경망이 비슷하지만, CNN신경망은 특히나 가변점이 많음으로, 다양한 파라메터로 실험 필요
    @staticmethod
    def get_network_head(inp):
        output = Conv2D(256, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(inp)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(128, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(64, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Conv2D(32, kernel_size=(1, 5),
            padding='same', activation='sigmoid',
            kernel_initializer='random_normal')(output)
        output = BatchNormalization()(output)
        output = MaxPooling2D(pool_size=(1, 2))(output)
        output = Dropout(0.1)(output)
        output = Flatten()(output)
        return Model(inp, output)

    # 2차원 합성곱 신경망이라서 (배치크기, 스텝(2차원 크기), 자질벡터 크기, 1) 모양으로 학습 데이터 사용
    # 맨뒤에 1은 사실 - 보통 합성곱 신경망은 이미지 데이터 취급을 위해서, R,G,B와 같은 값이 마지막 차원에서 사용되는데
    # 주식데이터는 이런 채널값이 없음으로 1로 고정(하드코딩)
    def train_on_batch(self, x, y):
        x = np.array(x).reshape((-1, self.num_steps, self.input_dim, 1))
        return super().train_on_batch(x, y)

    def predict(self, sample):
        sample = np.array(sample).reshape(
            (-1, self.num_steps, self.input_dim, 1))
        return super().predict(sample)
