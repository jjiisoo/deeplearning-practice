# # pandas로 데이터 열어보기 - "무슨 데이터인지"
import pandas as pd
import os
import numpy as np


DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "수업용데이터")


def path(file):
    return os.path.join(DATA, file)


# # BASE = "/Users/jisu/Documents/GitHub/deeplearning-practice/수업용데이터/"

# # df = pd.read_csv(
# #     BASE + "11_설비센서_ai4i.csv",
# #     encoding="utf-8-sig",
# # )
# df = pd.read_csv(
#     path(
#         "11_설비센서_ai4i.csv",
#     ),
#     encoding="utf-8-sig",
# )
# print(df.shape)
# print(list(df.columns))
# print(df.info())
# print(df.dtypes)
# print(df.describe(include="O"))
# print(df.isnull().sum())
# # 고장 4%

# # pandas 기본 동작
# print(df["공정온도"].head(3))
# err = df[df["고장여부"] == 1]
# print(round(err["고장여부"].mean(), 2), round(err["공구마모"].mean(), 2))
# print(err["공구마모"])


# nu = df.drop(columns=["설비ID", "타입"])
# print(nu.corr().round(2))

# import matplotlib.pyplot as plt
# import seaborn as sns

# # macOS 기본 한글 폰트 설정
# plt.rc("font", family="AppleGothic")
# plt.rcParams["axes.unicode_minus"] = False

# # sns.heatmap(nu.corr(), annot=True, fmt=".2f")
# # plt.title("변수 간 상관관계 히트맵")
# # plt.show()

# fake = df.copy()
# fake["설비번호"] = range(1, len(df) + 1)
# fake["번호_상관"] = fake["설비번호"] * 0 + fake["공기온도"] * 2 + 10
# print(round(fake["공기온도"].corr(fake["번호_상관"]), 4))

# dirty = pd.read_csv(path("12_제조센서_전처리.csv"), encoding="utf-8-sig")
# print(dirty.shape)
# print(dirty.head())

# print(dirty.isnull().sum())
# print(dirty[dirty["회전수"] < 0])

# print(dirty.duplicated().sum())
# print(dirty.duplicated(keep=False).sum())
# print(dirty[dirty.duplicated(keep=False)])


# pandas에 .values 붙이면 numpy 배열이 된다

# 12 파일 읽어와서 진동 3.2 넘는 행 골라내기 / 몇행인지 그 행들의 측정 ID, 설비명, 진동, 상태 출력
df1 = pd.read_csv(path("12_제조센서_전처리.csv"), encoding="utf-8-sig")

q = df1.copy()
print(q[q["진동"] > 3.2])
