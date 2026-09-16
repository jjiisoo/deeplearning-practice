# =====================================================================
#  실습 2 — 진동으로 압연 모터 전류를 맞혀 보기 (02 다변수 선형회귀)
# =====================================================================
#  실행: python 실습2_문제.py      (이 파일이 있는 폴더에서)
#  필요한 것: numpy, pandas  /  데이터는 옆의 데이터/ 폴더에 들어 있습니다.
#
#  [상황]
#    P제철 열간압연기(SPM01)에 진동센서 두 개(TOP·BOT)와 모터 전류계가 붙어 있습니다.
#    그런데 전류계가 자주 고장 납니다. 진동만 있을 때 전류를 추정할 수 있을까요?
#    맞힐 대상(정답) = CUR-MTR_RMS  (모터 전류의 실효값)
#
#  [쓰는 도구]  02 에서 배운 것 전부. 새 라이브러리 없습니다.
#    다변수 X / train·test 분할 / 열별 표준화(학습용 통계로만) / 경사하강 / R2
#    ※ 02_선형회귀_다변수_train_test.py 를 옆에 띄워 놓고 베껴 쓰세요. 그게 정상입니다.
#
#  [푸는 법]  TODO 를 위에서부터 하나씩 채우고, 그때그때 실행해서 숫자를 확인하세요.
#             한 번에 다 짜고 실행하면 어디서 틀렸는지 못 찾습니다.
# =====================================================================

import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "데이터")
d = pd.read_csv(os.path.join(DATA, "T-CR1-SPM01_압연특징.csv"), encoding="utf-8-sig")
# ↑ encoding="utf-8-sig" 빠뜨리면 첫 열 이름이 깨져서 KeyError 납니다.


# =====================================================================
# A. 데이터부터 본다  (모델 얘기는 아직 이르다)
# =====================================================================
# [A1] 표의 모양과 열 이름, 결측 개수를 찍으세요.
#      힌트: d.shape / list(d.columns) / d.isna().sum().sum()
# TODO
print(d.shape)
print(list(d.columns))
print(d.isnull().sum().sum())

# [A2] 정답으로 쓸 CUR-MTR_RMS 의 요약통계를 보세요. (describe)
#      → 이 값이 대략 몇에서 몇 사이인지 말할 수 있어야 합니다.
#        나중에 "MSE 500" 이 큰 건지 작은 건지 판단하는 기준이 됩니다.
# TODO
print(d["CUR-MTR_RMS"].describe())

# [A3] 숫자 열들의 상관계수 중, CUR-MTR_RMS 와의 상관만 크기순으로 보세요.
#      힌트: d.corr(numeric_only=True)["CUR-MTR_RMS"].sort_values()
#
#      ★ 보고 나서 답하세요 (주석으로 적어 두기) ★
#        (1) 상관이 0.98, 0.99 로 말도 안 되게 높은 열이 몇 개 보입니다. 이름이 뭔가요?
#        (2) 그 열들을 입력으로 쓰면 안 되는 이유가 있습니다. 뭘까요?
#            힌트: 열 이름의 앞부분을 보세요. CUR-MTR-... 로 시작하죠.
#                 전류계가 고장 나서 전류를 추정하려는 건데, 그 입력은 어디서 옵니까?
#      내 답: (1) CUR-MTR_STD, CUR-MTR_PTP
#             (2) 진동
# TODO

print(d.corr(numeric_only=True)["CUR-MTR_RMS"].sort_values())
# =====================================================================
# B. 입력 고르고 train / test 나누기
# =====================================================================
# [B1] 입력(특징) 4개를 아래 이름 그대로 쓰세요. 정답은 CUR-MTR_RMS.
특징이름 = ["VIB-BOT_RMS", "VIB-BOT_PTP", "VIB-BOT_KUR", "VIB-TOP_RMS", "CUR-MTR_STD"]
# X = ...   (d[특징이름].values.astype(float))
# y = ...   (d["CUR-MTR_RMS"].values.astype(float))
# X.shape, y.shape 를 찍어서 (570, 4) 와 (570,) 인지 확인하세요.
# TODO

X = d[특징이름].values.astype(float)
y = d["CUR-MTR_RMS"].values.astype(float)
print(X.shape, y.shape)
# [B2] 7:3 으로 나누세요. 반드시 '섞은 다음에' 자릅니다.
#      RandomState(42) 를 쓰면 정답지와 숫자가 똑같이 나옵니다.
#      힌트: 순서 = np.random.RandomState(42).permutation(len(X))
#            n_train = int(len(X) * 0.7)
#      학습용 몇 대 / 시험용 몇 대인지 찍으세요.
# TODO
순서 = np.random.RandomState(42).permutation(len(X))
n_train = int(len(X) * 0.7)
tr, te = (
    순서[:n_train],
    순서[n_train:],
)

X_train, X_test = X[tr], X[te]
y_train, y_test = (
    y[tr],
    y[te],
)

print("\n[2] 학습용", len(tr), "대 / 시험용", len(te), "대")

# [B3] 열별 표준화. ★ mu 와 sd 는 학습용에서만 구합니다 ★
#      시험용도 학습용의 mu, sd 로 변환하세요.
#      확인: 표준화 후 학습용 열별 평균은 0, 퍼짐은 1.
#            시험용 평균은 0 이 아닙니다. 그게 맞습니다 (이유를 말할 수 있어야 합니다).
# TODO
mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)
Z_train = (X_train - mu) / sd
Z_test = (X_test - mu) / sd
print(np.abs(Z_train.mean(axis=0)).round(3))
print(Z_train.std(axis=0).round(3))


# =====================================================================
# C. 학습 — 02 의 함수를 그대로 가져다 쓰세요
# =====================================================================
# [C1] 예측 / 손실 / 기울기_밟아보기 / 학습 / R2 / MSE 를 02 에서 복사해 오세요.
#      한 글자도 안 바꿔도 됩니다. 그게 이 실습의 포인트입니다.
#      (데이터가 바뀌어도 걷는 방법은 안 바뀝니다)
# TODO
def 예측(Z, w, b):
    return Z @ w + b


def 손실(Z, y, w, b):  # 01 과 완전히 같음: (실제 − 예측)² 의 평균 = MSE
    return np.mean((y - 예측(Z, w, b)) ** 2)


h = 0.0001


def 기울기_밟아보기(Z, y, w, b):
    gw = np.zeros(len(w))
    for j in range(len(w)):  # j = 0,1,2,3 : j 번째 손잡이만 움직여 본다
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[j] += h  # j 번째만 살짝 키우고
        w_minus[j] -= h  # j 번째만 살짝 줄여서
        gw[j] = (손실(Z, y, w_plus, b) - 손실(Z, y, w_minus, b)) / (
            2 * h
        )  # (오른쪽 손실 − 왼쪽 손실) ÷ 거리
    gb = (손실(Z, y, w, b + h) - 손실(Z, y, w, b - h)) / (2 * h)  # b 도 한 번
    return gw, gb


def 학습(Z, y, lr=0.1, epochs=500):
    w = np.zeros(Z.shape[1])  # Z.shape[1] = 열 수 = 4. 가중치 4개를 0 에서 출발
    b = 0.0
    for _ in range(epochs):  # 500 바퀴 (에폭 500)
        gw, gb = 기울기_밟아보기(Z, y, w, b)
        w = w - lr * gw
        b = b - lr * gb
    return w, b


# [C2] 학습용으로 학습(lr=0.1, epochs=500)하고, 특징별 가중치와 절편 b 를 찍으세요.
# TODO

w, b = 학습(Z_train, y_train)
print("\n[4] 학습 완료 — 표준화 눈금의 가중치")
for 이름, wi in zip(특징이름, w):
    print(
        f"    {이름:6s} w = {wi:+.3f}"
    )  # {:+.3f} = 부호 붙여 소수 3자리 (00 미리보기 ①)
print(f"    절편 b = {b:.3f}")


# [C3] 학습용 R2 / 시험용 R2 를 나란히 찍으세요. MSE 도 같이.
#
#      ★ 답하세요 ★
#        차이가 얼마입니까? 02 의 경보선(0.05 정상 / 0.10 넘으면 의심)에 비춰 보면
#        이 모델은 건강한가요, 과적합인가요?
#        ai4i 데이터(02)에서는 차이가 0.03 이었습니다. 왜 여기선 다를까요?
#      내 답: 과적합이다. 데이터가 다르기 때문이다.
# TODO
def MSE(y, yhat):  # 손실과 같은 식. 채점용으로 이름만 따로
    return np.mean((y - yhat) ** 2)


def R2(y, yhat):  # 01 9번의 R². 1 에 가까울수록 좋음, 0 = 평균만 말하는 수준
    return 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)


tr_pred = 예측(Z_train, w, b)
te_pred = 예측(Z_test, w, b)
print("\n[5] 채점")
print(
    f"    학습용(train)  R² {R2(y_train, tr_pred):.4f}   손실 {MSE(y_train, tr_pred):.3f}"
)
print(
    f"    시험용(test)   R² {R2(y_test, te_pred):.4f}   손실 {MSE(y_test, te_pred):.3f}"
)
# =====================================================================
# D. 함정 1 — "점수가 너무 좋으면 의심하라"
# =====================================================================
# [D1] 특징에 "CUR-MTR_STD" 를 하나 추가해서(총 5개) 다시 학습하고 채점하세요.
#      B2 의 분할(순서)은 그대로 재사용합니다. 표준화는 다시 해야 합니다(열이 5개니까).
#
#      ★ 답하세요 ★
#        (1) R2 가 몇으로 나왔나요? 학습용·시험용 둘 다 적으세요.
#        (2) 이 모델을 현장에 넣으면 잘 될까요? 이유는?
#        (3) 이걸 부르는 이름이 있습니다 — '누수(leakage)'.
#            이 경우 정확히 무엇이 새어 들어온 겁니까?
#      내 답: (1) 둘다 약 1로 나왔다.
#             (2) 현장에 넣으면 잘 안 된다. R2는 좋지만 손실도 크고 정답이 이미 학습되어서 성능이 높은 것이다.
#             (3) 정답이 새어 들어온 것이다.
# TODO

누수특징이름 = 특징이름 + ["CUR-MTR_STD"]

X_leak = d[누수특징이름].values.astype(float)

X_leak_train = X_leak[tr]
X_leak_test = X_leak[te]

mu_leak = X_leak_train.mean(axis=0)
sd_leak = X_leak_train.std(axis=0)

Z_leak_train = (X_leak_train - mu_leak) / sd_leak
Z_leak_test = (X_leak_test - mu_leak) / sd_leak

w_leak, b_leak = 학습(
    Z_leak_train,
    y_train,
    lr=0.1,
    epochs=500,
)

tr_pred_leak = 예측(Z_leak_train, w_leak, b_leak)
te_pred_leak = 예측(Z_leak_test, w_leak, b_leak)

print(
    f"    학습용(train)  R² {R2(y_train, tr_pred_leak):.4f}   "
    f"손실 {MSE(y_train, tr_pred_leak):.3f}"
)
print(
    f"    시험용(test)   R² {R2(y_test, te_pred_leak):.4f}   "
    f"손실 {MSE(y_test, te_pred_leak):.3f}"
)
# =====================================================================
# E. 함정 2 — 상관 순위와 실제 쓸모는 다르다
# =====================================================================
# [E1] 02 §6-1 처럼 특징을 하나씩 빼고 다시 학습해, 학습용·시험용 R2 를 각각 찍으세요.
#      (4개 특징이니 4줄이 나옵니다. 힌트: Z_train[:, 남길])
#
#      ★ 먼저 예상하고 적으세요. 실행은 그다음에. ★
#        A3 에서 본 상관을 보면 VIB-BOT_RMS 가 0.74 로 1등,
#        VIB-TOP_RMS 는 0.09 로 사실상 무관해 보입니다.
#        그럼 VIB-TOP_RMS 를 빼도 점수가 안 변하겠죠?
#      내 예상: 상관이 낮아 점수가 거의 변하지 않을 것 같다.
# TODO
total_test_r2 = R2(y_test, te_pred)

for num, name in enumerate(특징이름):
    남길 = [i for i in range(len(특징이름)) if i != num]

    Ztr = Z_train[:, 남길]
    Zte = Z_test[:, 남길]

    w_temp, b_temp = 학습(
        Ztr,
        y_train,
        lr=0.1,
        epochs=500,
    )

    pred_train = 예측(Ztr, w_temp, b_temp)
    pred_test = 예측(Zte, w_temp, b_temp)

    print(
        f"    {name:12s} 제거 | "
        f"train R² {R2(y_train, pred_train):.4f} | "
        f"test R² {R2(y_test, pred_test):.4f}"
    )


# [E2] 실행 결과를 보고 답하세요.
#        (1) 예상이 맞았나요?
#        (2) VIB-BOT_RMS 를 뺐을 때 시험용 점수가 어떻게 됐습니까?
#            왜 그럴까요?  힌트: d[특징이름].corr() 를 찍어 보세요.
#                                 VIB-BOT_RMS 와 VIB-BOT_PTP 의 상관은?
#        (3) VIB-TOP_RMS 를 뺐을 때는요? 정답과 상관이 0.09 밖에 안 되는데 왜?
#        (4) 여기서 얻을 교훈을 한 줄로 적으세요.
#      내 답: 예상과 다르다,  0.953446의 상관관계를 가진다. test R² 0.9976, test R² 0.9982, 정답과의 상관계수만 보고 특징을 뺄지 판단하면 안된다.
# TODO
print(d[특징이름].corr())

# =====================================================================
# F. 함정 3 — 섞어서 자른 게 정말 옳았나
# =====================================================================
# [F1] 이 데이터는 MEAS_DT(측정시각) 순으로 정렬돼 있습니다. 1월부터 12월까지.
#      02 에서는 "섞고 잘라라, 안 섞으면 편향된다" 고 배웠죠.
#      이번엔 반대로 해 보세요 — 섞지 말고 앞 399행을 학습용, 뒤 171행을 시험용으로.
#      (즉 1~9월로 배워서 10~12월을 맞히기)
#
#      ★ 답하세요 ★
#        (1) 시험용 R2 가 섞었을 때(C3)보다 높나요, 낮나요?
#        (2) 결과가 예상과 다를 겁니다. 그래도 실무에서 예측 모델을 만들 때는
#            보통 이 '시간순 분할' 쪽을 씁니다. 왜 그럴까요?
#            힌트: 현장에 배포된 모델이 맞혀야 하는 데이터는 '언제' 것입니까?
#      내 답: (1) 높다.
#             (2) 미래 데이터를 예측해야한다, 과거 데이터로 학습하고 미래 데이터를 판단해야하기 때문이다.
# TODO
n_train = 399

X_train = X[:n_train]
X_test = X[n_train:]

y_train = y[:n_train]
y_test = y[n_train:]

mu = X_train.mean(axis=0)
sd = X_train.std(axis=0)

Z_rain = (X_train - mu) / sd
Z_test = (X_test - mu) / sd

w_time, b_time = 학습(
    Z_train,
    y_train,
    lr=0.1,
    epochs=500,
)

train_pred = 예측(Z_train, w_time, b_time)
test_pred = 예측(Z_test, w_time, b_time)

random_test_r2 = R2(y_test, te_pred)
test_r2 = R2(y_test, test_pred)

print(
    f"    학습용(train) R² {R2(y_train, train_pred):.4f}   "
    f"손실 {MSE(y_train, train_pred):.3f}"
)
print(f"    시험용(test)  R² {test_r2:.4f}   손실 {MSE(y_test, test_pred):.3f}")
print(f"    무작위 분할 test R²: {random_test_r2:.4f}")
print(f"    시간순 분할 test R²: {test_r2:.4f}")


# =====================================================================
# G. 데이터가 몇 대면 충분한가
# =====================================================================
# [G1] 학습용을 5 / 10 / 30 / 100 / 399 대로 바꿔 가며 학습하고,
#      학습용 R2 와 시험용 R2 를 표처럼 찍으세요. (적은 데이터는 epochs 를 늘리세요)
#
#      ★ 답하세요 ★
#        (1) 시험용 R2 가 음수로 나오는 구간이 있습니다. 음수는 무슨 뜻입니까?
#            힌트: R2 = 0 이 '무조건 평균만 대답하는 모델' 입니다.
#        (2) 데이터가 늘 때 학습용 점수와 시험용 점수는 각각 어느 방향으로 움직입니까?
#        (3) 이 설비에서 쓸 만한 모델을 만들려면 최소 몇 건쯤 필요해 보입니까?
#      내 답: 음수는 성능이 많이 나쁘다는 뜻이다. 평균만 예측하는 모델보다도 더 떨어진 성능
#           데이터가 늘어나면 학습용 r2와 시험용 r2의 차이가 줄어든다
#           100건 이상
# TODO
train_data = [5, 10, 30, 100, 399]

X_test = X[399:]
y_test = y[399:]


for n in train_data:
    X = X[:n]
    y = y[:n]

    mu = X.mean(axis=0)
    sd = X.std(axis=0)

    sd[sd == 0] = 1

    Z_small_train = (X - mu) / sd
    Z_small_test = (X_test - mu) / sd

    epochs = 2000 if n <= 30 else 500

    w_small, b_small = 학습(
        Z_small_train,
        y,
        lr=0.1,
        epochs=epochs,
    )

    small_train_pred = 예측(Z_small_train, w_small, b_small)
    small_test_pred = 예측(Z_small_test, w_small, b_small)

    train_score = R2(y, small_train_pred)
    test_score = R2(y_test, small_test_pred)

    print(f"{n:9d} | {train_score:8.4f} | {test_score:7.4f}")

# =====================================================================
# H. 마무리 — 보고서 3줄
# =====================================================================
# 팀장에게 보고한다고 치고, 아래 세 줄을 채우세요.
#
#   1) 전류계가 고장 났을 때 진동으로 전류를 추정할 수 있는가? (된다/안 된다/조건부)
#      근거 점수: 조건부, 시간순 분할에서 출력된 시험용 R2, MSE
#
#   2) 이 모델을 쓸 때 반드시 붙여야 할 경고 문구 한 줄: 전류계에서 만들어지는 값을 입력으로 사용하면 안된다.
#
#   3) 점수를 더 올리려면 다음에 뭘 해 보겠는가? (한 가지만, 이유와 함께) : 다른 조건의 데이터들을 더 수집한다. 일반화 성능을 높이기 위해 계절 같이 시간순 평가에 도움되는 데이터를 추가로 수집해 성능을 높이기 위해서다.
#
# =====================================================================
