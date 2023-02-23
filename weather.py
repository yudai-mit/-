import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#データ読み込み
weather=pd.read_csv("weather.csv")
df=pd.DataFrame(weather)
del df["year/month/day"]

df0=df[df["target"]==0]
df1=df[df["target"]==1]
#グラフにプロット
"""plt.figure(figsize=(5,5))
plt.scatter(df0["cloud"],df0["suntime"],color="b",alpha=0.5)
plt.scatter(df1["cloud"],df1["suntime"],color="r",alpha=0.5)
plt.show()"""
#データをトレーニング用とテスト用に分ける
X=df[["cloud","suntime"]]

x_train,x_test,y_train,y_test=train_test_split(X,df["target"],random_state=0)

#トレーニング用についての処理
"""
df_train=pd.DataFrame(x_train)
df_train["target"]=y_train

df0_train=df_train[df_train["target"]==0]
df1_train=df_train[df_train["target"]==1]

plt.figure(figsize=(5,5))
plt.scatter(df0_train["cloud"],df0_train["suntime"],color="b",alpha=0.5)
plt.scatter(df1_train["cloud"],df1_train["suntime"],color="r",alpha=0.5)
plt.title("train:75%")
plt.show()
"""

#テスト用についての処理
"""
df_test=pd.DataFrame(x_test)

df_test["target"]=y_test

df0_test=df_test[df_test["target"]==0]
df1_test=df_test[df_test["target"]==1]


plt.figure(figsize=(5,5))
plt.scatter(df0_test["cloud"],df0_test["suntime"],color="b",alpha=0.5)
plt.scatter(df1_test["cloud"],df1_test["suntime"],color="r",alpha=0.5)
plt.title("train:25%")
plt.show()
"""

#機械への学習処理
model=svm.SVC()
model.fit(x_train,y_train)
print(x_test)
#機械へのテスト

pred=model.predict(x_test)
df_test=pd.DataFrame(x_test)
df_test["target"]=pred

df0_pred=df_test[df_test["target"]==0]
df1_pred=df_test[df_test["target"]==1]

"""
plt.figure(figsize=(5,5))
plt.scatter(df0_pred["cloud"],df0_pred["suntime"],color="b",alpha=0.5)
plt.scatter(df1_pred["cloud"],df1_pred["suntime"],color="r",alpha=0.5)
plt.title("predict")
plt.show()
"""
"""
#正解率の出力
pred=model.predict(x_test)
print(x_test)
score=accuracy_score(y_test,pred)
print("正解率：",score*100,"%")
"""
pred=model.predict([[0,1]])
print("0,1,=",pred)

plt.figure(figsize=(5,5))
plt.scatter(df0_pred["cloud"],df0_pred["suntime"],color="b",alpha=0.5)
plt.scatter(df1_pred["cloud"],df1_pred["suntime"],color="r",alpha=0.5)
plt.scatter([10],[5],color="r",marker='x',s=300)
plt.title("predict")
plt.show()
