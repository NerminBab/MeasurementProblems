
# RATING PRODUCTS

"""
Kullanıcı ve zaman ağırlıklı kurs puanı hesaplama
(50+ saat) Python A-Z: Veri Bilimi ve Machine Learning
Puan: 4.8 (4.764925)
Toplam Puan: 4611
Puan Yüzdeleri: 75, 20, 4, 1, <1
Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

"""

# kütüphane importlarını yapalım:
import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("dataset/course_reviews.csv")
df.head()
# verisetinde; kişilerin puanları, üye olma tarihi, sorduğu ve aldığı cevapların yüzdesi bulunmakta

df.shape

# rating dağılımı:
df["Rating"].value_counts()

df["Questions Asked"].value_counts() #3867 kişi soru sormamış

# sorulan soru kırılımında verdikleri ortalama puan ne kadardır?
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})

# AVERAGE
df["Rating"].mean()
# puan ortalam puanını hesapladık. ancak memnuniyeti de değerlendirmeliyiz bunu da puana yansıtmalıyız.

# 1. TIME-BASED WEIGHTED AVERAGE
# Puan zamanlarına göre ağırlıklı ortalama hesaplarsak daha doğru bir puanlama yapmış oluruz.
df.info()
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# yorumları gün cinsinden almalıyız.
current_date = pd.to_datetime("2021-02-10 0:0:0")

df["days"] = (current_date - df["Timestamp"]).dt.days
df.head()

# son 30 günde yapılan yorumlara erişelim:
df[df["days"] <= 30].count()

# son 30 gündeki puanların ortalamasına bakalım:
df.loc[df["days"] <= 30, "Rating"].mean()

# son 30-90 arasındaki günlerin puan ortalamalrına bakalım:
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

# 90-180 arasındaki günlerin puan ortalamalrına bakalım:
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

df.loc[df["days"] > 180, "Rating"].mean()
# son zamanlarda kursun puanı artıyor.

# zamana göre ortalama puan hesaplayalım:
df.loc[df["days"] <= 30, "Rating"].mean() * 28/100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26/100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24/100 + \
df.loc[df["days"] > 180, "Rating"].mean() * 22/100

def time_based_weigted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1/100 + \
           dataframe.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * w2/100 + \
           dataframe.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * w3/100 + \
           dataframe.loc[df["days"] > 180, "Rating"].mean() * w4/100
time_based_weigted_average(df)

time_based_weigted_average(df, 30, 26, 22, 22)



# USER-BASED WEIGHTED AVERAGE
# her kullanıcının verdiği puanın ağırlığı aynı mı olmalı? kursun %10 ile %100 ünü izleyen kullanıcının verdiği puanların değeri aynı değildir.
df.head()
df.groupby("Progress").agg({"Rating": "mean"})
# kurstaki ilerleme durumu ile verilen puanların arttığı gözlemlendi.
df.loc[df["Progress"] <= 10, "Rating"].mean() * 22/100 + \
df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24/100 + \
df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26/100 + \
df.loc[df["Progress"] > 75, "Rating"].mean() * 28/100

def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <=10, "Rating"].mean() * w1/100 + \
           dataframe.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[df["Progress"] > 75, "Rating"].mean() * w4 / 100
user_based_weighted_average(df, 20, 24, 26, 30)

# zamana göre hasssaslaştırdık.


# WEIGHTED RATING
# time_based ve user_based in ağırlıklı ortalamasını alarak hesapayacağız:
def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weigted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w/100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60) # ilerleme yüzdelerine göre kullanıcıların verdiği puanları önceliklendirdik
