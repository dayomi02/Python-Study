import pandas as pd     # 데이터 조작과 분석을 위한 라이브러리
import numpy as np      # 수치 연산을 위한 라이브러리
import random           # 난수 생성과 관련된 기능을 제공 (파이썬 내장 라이브러리)
import os               # 운영 체제와 상호작용하기 위한 기능을 제공 (파이썬 내장 라이브러리)

def seed_everything(seed):                    # 난수 시드를 고정하는 함수
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정


train = pd.read_csv('./open/train.csv')
test = pd.read_csv('./open/test.csv')
train.head()
test.head()


# train_x는 독립변수이므로 종속변수(풍속 (m/s))를 제거합니다.
# 또한 target 이외의 분석에 활용하지 않는 데이터(id)를 제거합니다.
train_x = train.drop(columns=['ID', '풍속 (m/s)'], axis = 1)

# train_y는 종속변수로 값을 설정합니다.
train_y = train['풍속 (m/s)']

# train에서와 마찬가지로 분석에 활용하지 않는 데이터(id)를 제거합니다.
test_x = test.drop(columns=['ID'])


from sklearn.preprocessing import LabelEncoder  # sklearn.preprocessing 모듈에서 LabelEncoder를 임포트
                                                # LabelEncoder는 범주형 변수를 숫자로 인코딩하는데 사용

le = LabelEncoder()                                            # LabelEncoder 객체를 생성
le = le.fit(train_x['측정 시간대'])                            # '측정 시간대' 열을 기준으로 LabelEncoder를 학습
train_x['측정 시간대'] = le.transform(train_x['측정 시간대'])  # 학습된 LabelEncoder 객체를 사용하여 '측정 시간대' 열의 값을 숫자로 변환
test_x['측정 시간대'] = le.transform(test_x['측정 시간대'])


# 추가
def nmae(
    X_val, y_val, estimator, labels,
    X_train, y_train, weight_val=None, weight_train=None,
    *args, **kwargs
):
    def _nmae(true, pred):
        mae = np.mean(np.abs(true-pred))
        score = mae / np.mean(np.abs(true))
        return score

    y_pred = estimator.predict(X_val)
    val_loss = _nmae(y_val, y_pred)
    y_pred = estimator.predict(X_train)
    train_loss = _nmae(y_train, y_pred)

    return val_loss, {
        "val_loss": val_loss,
        "train_loss": train_loss,
    }

from flaml import AutoML

model = AutoML()

# Model Train
model.fit(train_x, train_y, task='regression', time_budget=60*2, metric=nmae, ensemble=True, early_stop=True, seed=42)

# Predict
pred_y = model.predict(test_x)


submission = pd.read_csv('./open/sample_submission.csv')
submission['풍속 (m/s)'] = pred_y
submission.head()
submission.to_csv('20230724.csv', index=  False)
