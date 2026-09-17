import pandas as pd
import os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "수업용데이터")


# 현재 폴더의 상위 폴더 / 수업용데이터
def path(file):
    return os.path.join(DATA, file)


df = pd.read_csv(path("11_설비센서_ai4i.csv"), encoding="utf-8-sig")


# =====================================================================
#  00_pandas 복습
# =====================================================================

print(df.shape)
print(df.info())
print(df.describe())
print(df.describe(include="O"))
print(df.isnull().sum())
print(df.duplicated(keep=False).sum())

print(df["공정온도"].head())
error = df[df["고장여부"] == 1]
print(round(error["고장여부"].mean(), 2), round(error["공구마모"].mean(), 2))
print(error["공구마모"])

col = df.drop(columns=["설비ID", "타입"])  # 수치형 데이터만 남기기
print(col.corr().round(2))

# 히트맵 시각화
# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.rc("font", family="AppleGothic")
# plt.rcParams["axes.unicode_minus"] = False

# sns.heatmap(nu.corr(), annot=True, fmt=".2f")
# plt.title("변수 간 상관관계 히트맵")
# plt.show()

df2 = df.copy()
df2["설비번호"] = range(1, len(df) + 1)  # 새로운 열 추가
df2["번호_상관"] = df2["설비번호"] * 0 + df2["공기온도"] * 2 + 10
print(df2.head(2))
print(round(df2["공기온도"].corr(df2["번호_상관"]), 4))  # 무조건 1나옴

dirty = pd.read_csv(path("12_제조센서_전처리.csv"), encoding="utf-8-sig")
print(dirty.shape)
print(dirty.info())
print(dirty.head(2))

print(dirty.isnull().sum())
print(dirty.duplicated().sum())
print(dirty[dirty.duplicated(keep=False)])
dirty = dirty.drop_duplicates()  # keep = "first" : default / keep = "last"
print(dirty.duplicated().sum())

print(dirty[dirty["진동"] > 3.2])  # 진동이 3.2 초과인 행 출력하기
print((dirty["진동"] > (3.2)).sum())


# =====================================================================
#  01_numpy 복습
# =====================================================================
# 공기온도로 공정온도 예측하기
x = df["공기온도"].values.astype(float)  # array로 변경 / 입력값
y = df["공정온도"].values.astype(float)  # 정답값

print("데이터 개수", len(x))
# print(x.describe()) 이렇게 사용하면 에러남. 위에서 numpy array로 바꿨기 때문
print("x 범위", x.min(), "~", x.max())
print("y 범위", y.min(), "~", y.max())


def pred(x, w, b):
    return w * x + b


print(pred(x, 1.0, 10.0)[:3])  # 예측한 값


def loss(x, y, w, b):
    error = y - pred(x, w, b)
    return np.mean(error**2).round(4)  # MSE


print(loss(x, y, 1.0, 10.0))

h = 0.0001


def gradient(x, y, w, b):
    gw = (loss(x, y, w + h, b) - loss(x, y, w - h, b)) / (2 * h)
    gb = (loss(x, y, w, b + h) - loss(x, y, w, b - h)) / (2 * h)
    return gw, gb


# w, b를 좌우로 조금씩 움직여 손실 변화를 확인한다. 손실이 가장 빠르게 증가하는 방향 반환
# 학습할 때는 이와는 반대 방향으로 w, b 수정

gw, gb = gradient(x, y, 1.0, 9.0)
print(gw.round(4))
print(gb.round(4))
# 구한 기울기의 반대 방향으로 움직인다.
lr = 0.00001
w, b = 1.0, 9.0
print("이전", loss(x, y, w, b))
w = w - lr * gw
b = b - lr * gb
print("이후", loss(x, y, w, b))


with np.errstate(over="ignore", invalid="ignore"):  # numpy의 특정 경고 무시
    for epoch in range(20):
        gw, gb = gradient(x, y, w, b)
        w, b = w - 0.001 * gw, b - 0.001 * gb

        # print(
        #     epoch,
        #     "w:",
        #     w,
        #     "b:",
        #     b,
        #     "loss:",
        #     loss(x, y, w, b),
        # )

print(loss(x, y, w, b))


# 표준화
m = x.mean()
s = x.std()
z = (x - m) / s


def train(z, y, lr=0.1, epochs=300, ap=True):
    w, b = 0.0, 0.0
    for epoch in range(epochs):
        gw, gb = gradient(z, y, w, b)
        w = w - lr * gw
        b = b - lr * gb
        if ap and epoch in (0, 1, 5, 10, 100, 299):
            print(f"epoch {epoch:3d} loss {loss(z, y, w, b):10.4f}")
    return w, b


w_z, b_z = train(z, y)

w_gd = w_z / s
b_gd = b_z - w_z * m / s
print(f"loss {loss(x, y, w_gd, b_gd):.4f}")


# =====================================================================
#  02_선형회귀 복습
# =====================================================================

features = ["공기온도", "회전수", "토크", "공구마모"]
X = df[features].values.astype(float)
y = df["공정온도"].values.astype(float)

print(X.shape, y.shape)

rng = np.random.RandomState(42)
order = rng.permutation(len(X))
n_train = int(len(X) * 0.7)
tr, te = (order[:n_train], order[n_train:])

X_train, X_test = X[tr], X[te]
y_train, y_test = y[tr], y[te]

print(len(tr), len(te))
