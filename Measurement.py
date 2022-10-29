# Rating Products
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

# Average
df["Rating"].mean()
# puan ortalam puanını hesapladık. ancak memnuniyeti de değerlendirmeliyiz bunu da puana yansıtmalıyız.

# Time-Based Weighted Average
# puan zamanlarına göre ağırlıklı ortalama hesaplarsak daha doğru bir puanlama yapmış oluruz.






#User-Based Weighted Average





# Weighted Rating