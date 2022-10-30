"""
* RATING PRODUCTS:
- Average
- Time-Based Weighted Average
- User-Based Weighted Average
- Weighted Rating
- Bayesian Average Rating Score

* SORTING PRODUCTS:
- Sorting by Rating
- Sorting by Comment Count or Purchase Count
- Sorting by Rating, Comment and Purchase
- Sorting by Bayesian Average Rating Score (Sorting Products with 5 star Rated)
- Hybrid Sorting: Bar Score + Diğer Faktörler


 * SORTING REVIEWS:
- Up-Down Difference Score
- Average Rating
- Wilson Lower Bound Score

""""

"""
Kullanıcı ve zaman ağırlıklı kurs puanı hesaplama
(50+ saat) Python A-Z: Veri Bilimi ve Machine Learning
Puan: 4.8 (4.764925)
Toplam Puan: 4611
Puan Yüzdeleri: 75, 20, 4, 1, <1
Yaklaşık Sayısal Karşılıkları: 3458, 922, 184, 46, 6

"""
# ****************************** RATING PRODUCTS **************************************************
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

df["Questions Asked"].value_counts()  # 3867 kişi soru sormamış

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
df.loc[df["days"] <= 30, "Rating"].mean() * 28 / 100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26 / 100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24 / 100 + \
df.loc[df["days"] > 180, "Rating"].mean() * 22 / 100


def time_based_weigted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[df["days"] > 180, "Rating"].mean() * w4 / 100


time_based_weigted_average(df)

time_based_weigted_average(df, 30, 26, 22, 22)

# 2. USER-BASED WEIGHTED AVERAGE
# her kullanıcının verdiği puanın ağırlığı aynı mı olmalı? kursun %10 ile %100 ünü izleyen kullanıcının verdiği puanların değeri aynı değildir.
df.head()
df.groupby("Progress").agg({"Rating": "mean"})
# kurstaki ilerleme durumu ile verilen puanların arttığı gözlemlendi.
df.loc[df["Progress"] <= 10, "Rating"].mean() * 22 / 100 + \
df.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * 24 / 100 + \
df.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * 26 / 100 + \
df.loc[df["Progress"] > 75, "Rating"].mean() * 28 / 100


def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           dataframe.loc[(df["Progress"] > 10) & (df["Progress"] <= 45), "Rating"].mean() * w2 / 100 + \
           dataframe.loc[(df["Progress"] > 45) & (df["Progress"] <= 75), "Rating"].mean() * w3 / 100 + \
           dataframe.loc[df["Progress"] > 75, "Rating"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)


# zamana göre hasssaslaştırdık.


# 3. WEIGHTED RATING
# time_based ve user_based in ağırlıklı ortalamasını alarak hesapayacağız:
def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weigted_average(dataframe) * time_w / 100 + user_based_weighted_average(dataframe) * user_w / 100


course_weighted_rating(df)

course_weighted_rating(df, time_w=40,
                       user_w=60)  # ilerleme yüzdelerine göre kullanıcıların verdiği puanları önceliklendirdik

# *************************************** SORTING PRODUCTS **************************************************
# SORTING BY RATING
# Kurs Sıralama:
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import math

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("dataset/product_sorting.csv")
df.head()
df.shape

# rating e bakarak sıralama yapmak rasyonel sonuçlar verir mi?
df.sort_values("rating", ascending=False).head(20)  # satın alma ve yorum sayısı gözardı edilmiş oldu

# SORTING BY COMMENT COUNT OR PURCHACE COUNT
df.sort_values("purchase_count", ascending=False).head(20)  # yorum sayısı olmadan gerçekçi olmaz sıralama

df.sort_values("commment_count", ascending=False).head(20)  # ücretsiz olarak da dağılmış olabilir.

# SORTING BY RATING, COMMENT and PURCHACE
# satınalma değişkenleri de rating gibi 1-5 arası değerlere çevirelim:
df["purchase_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["purchase_count"]]). \
    transform(df[["purchase_count"]])
df.head()

df.describe().T

# yorum değişkenleri de rating gibi 1-5 arası değerlere çevirelim:
df["comment_count_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["commment_count"]]). \
    transform(df[["commment_count"]])
df.head()

df.describe().T

# tüm değişkenleri dikkate alarak önem sıramızı belirleyip hassaslaştırabiliriz, sosyal ispat önemli olduğu için en yüksek puanı ona vereceğiz:
(df["comment_count_scaled"] * 32 / 100 +
 df["purchase_count_scaled"] * 26 / 100 +
 df["rating"] * 42 / 100)  # böylece skorlarımız hazır


def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 / 100 +
            dataframe["purchase_count_scaled"] * w2 / 100 +
            dataframe["rating"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)

df.sort_values("weighted_sorting_score", ascending=False).head(20)

# sadece veri bilimi kurslarını getirelim:
df[df["course_name"].str.contains("Veri Bilimi")].sort_values("weighted_sorting_score", ascending=False).head(20)


# BAYESIAN AVERAGE RATING SCORE
# Sorting products with 5 star rated (sorting products according to distribution of 5 star rating)

# puan dağılımları üzerinden olasılıksal bir ortalama yapar:

def bayesian_average_rating(n, confidence=0.95):
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


df["bar_score"] = df.apply(
    lambda x: bayesian_average_rating(x[["1_point", "2_point", "3_point", "4_point", "5_point"]]), axis=1)


df.sort_values("weighted_sorting_score", ascending=False).head(20)
df.sort_values("bar_score", ascending=False).head(20)

df[df["course_name"].index.isin([5, 1])]

df[df["course_name"].index.isin([5, 1])].sort_values("bar_score", ascending=False) # ürün commentin yüksek olmasına rağmen sıralamada düşük olduğunu gördük.



# KARMA SIRALAMA (HYBRID SORTING) BAR Score + Diğer Faktörler
def hybrid_sorting_score(dataframe, bar_w=60, wss_w=40):
    bar_score = dataframe.apply(lambda x: bayesian_average_rating(x[["1_point", "2_point", "3_point", "4_point", "5_point"]]), axis=1)
    wss_score = weighted_sorting_score(dataframe)
    return bar_score * bar_w / 100 + wss_score * wss_w /100

df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values("hybrid_sorting_score", ascending=False).head(20)

df[df["course_name"].str.contains("Veri Bilimi")].sort_values("hybrid_sorting_score", ascending=False).head(20)


# ***************************** SORTING REVİEWS ****************************************************
# SORTING REVIEWS

# Yorumların sıralanması verilen puandan bağımsızdır, çünkü önemli olan social proof un yansıtılmasıdır. İş bilgisinin yanında user quality score da dikkate alınır.

# UP-DOWN DIFFERENCE SCORE = up rating - down rating:
# Review 1: 600 up, 400 down, total 1000
# Review 2: 5500 up, 4500 down, total 10000

def score_up_down_diff(up,down):
    return up-down

# Review 1:
score_up_down_diff(600, 400)

# Review 2:
score_up_down_diff(5500, 4500)

# Yüzdeleri hesaba bakılmadığı için bu yöntem eksiktir, yanlılık barındıdır.


# AVERAGE RATING = up rating / all rating:
def score_average_rating(up, down):
    if up+down == 0:
        return 0
    return up/(up+down)

score_average_rating(600, 400)
score_average_rating(5500, 4500)
# up-down difference score hesabından farklı ancak daha akla yatkın

#yeni bir senaryo üzerinden gidelim:
# Review 1: 2 up, 0 down, total 2
# Review 2: 100 up, 1 down, total 101

score_average_rating(2, 0)
score_average_rating(100,1)
# bu yöntem de frekans bilgilerini dikkate almadığı için yetersiz kalmıştır.



# WILSON ALT SINIR PUANI (WILSON LOWER BOUND SCORE):
# Bu yöntem ikili interaction barındıran durumları skorlamayı sağlar. Olayın gözlenmesi ile ilgili olasılık dağılımı
# Olasılık ile örneklem içinden genelleme yaparak kitleye yansıtabiliriz.

# bir önceki örenekte 600 up, 400 down örneğinde up oranı 0.6 dır, güven ararlığı 0.5-0.7 aralığındadır.
# istatistiksel olarak 100 kullanıcıdan 95 inin %5 hata payı olmakla birlikte yorumun up oranı 0.5 ile 0.7 arasında olacaktır.

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score Hesaplama
    -Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    -Hesaplanacak skor ürün sıralaması için kullanılır.
    :param up: int
    :param down: int
    :param confidence: float
    :return: wilson score: float
    """
    n = up+down
    if n == 0:
        return 0
    z = st.norm.ppf(1-(1-confidence)/2)
    phat = 1.0*up/n
    return (phat+z*z/(2*n)-z*math.sqrt((phat*(1-phat)+z*z/(4*n))/n))/(1+z*z/n)

wilson_lower_bound(600, 400)
wilson_lower_bound(5500, 4500)

wilson_lower_bound(2, 0)
wilson_lower_bound(100, 1)
# Bu sonuçlar akla daha yatkın. %95 güven ve %5 hata payı ile hesaplamış olduk.

# NOT: Eğer skorlar 1-5 arasındaysa 1-3 arasına negatif, 4-5 e pozitif değer verilerek binary hale getirilip bernoulliye uygun şekle getirilip WLB uygulanabilir. Ancak böyleyapmak beraberinde bazı problemleri getirebilir bu sebeple bayesian average rating yapmak daha uygun olur.

