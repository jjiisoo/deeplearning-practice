import pandas as pd
import os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "수업용데이터")


# 현재 폴더의 상위 폴더 / 수업용데이터
def path(file):
    return os.path.join(DATA, file)


df = pd.read_csv(path("11_설비센서_ai4i.csv"), encoding="utf-8-sig")

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
