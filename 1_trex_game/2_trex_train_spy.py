#pip install scikit-learn


import glob
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

#buradaki dense:  "Fully Connected" veya "Dense" katmanını temsil eder.  yani; Bu katman, önceki katmandaki tüm birimlerin
#birbirine bağlı olduğu ve her bir birimin, bir sonraki katmandaki tüm birimlerle bağlantılı olduğu bir yapıya sahiptir. 
#Bu katman, girdi verilerini alır ve ağırlık matrisiyle çarparak çıktıyı üretir. 
#Ardından, genellikle bir aktivasyon fonksiyonuyla çıktı işlenir.

#dropout ise: seyreltme islemini temsil eder,

#flatten ise: duzlestirme islemi.

#conv2d ise: evrisim agimiz.

#MaxPooling2D ise: piksel ekleme kavramimizdir.


from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns




imgs = glob.glob("./img_nihai/*.png")

width = 125
height = 50

X = []
Y = []

#egitim oncesi bazi donusumler uyguliycaz resimlere,
for img in imgs:
    
    filename = os.path.basename(img)
    #burada _ ile split ettik ve 0. inseksi aldik bunun sebebi ise bizim resimlerimizin ilk basinda hangi hareket olduug yaziyor
    label = filename.split("_")[0]
    #simdi ise resmimizi convert ediyoruz yani size ini degistiriyoruz.
    im = np.array(Image.open(img).convert("L").resize((width, height)))
    im = im / 255
    X.append(im)
    Y.append(label)
    
X = np.array(X)
X = X.reshape(X.shape[0], width, height, 1)


#burada y nin icine yani down up ve right a one hot encode ve label encode islemini yapiyoruz.
#yani ilk once up down ve right i sayiya ceviriyoruz mesela 0 1 ve 2 olark sonrasinda one hot encoder yapiyopruz o da
#0 icin 100 1 icin 010 2 icin 001 seklinde sayilari birbirinden ayirt etmesini saglayan bir yontem.
def onehot_labels(values):
    label_encoder = LabelEncoder()
    #buradaki fit aslinda ne yapacagini ogreniyor gibi dusunebiliriz sonrasinda transform ediyor yani donusturuyor.
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse = False)
    #buranin sonunda 1 koyduk cunku integer_ encodede baktigimizda 169 gozukuyor sdece onu 1 yapmaliyiz ki hata almayalim.
    integer_encoded = integer_encoded.reshape(len(integer_encoded),1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return onehot_encoded

Y = onehot_labels(Y)
train_X, test_X, train_y, test_y = train_test_split(X, Y , test_size = 0.25, random_state = 2)



#cnn modelimizi insa ediyoruz.
#burasi layerlarimiizi uzerine ekleyecegimiz temel yapi.
model = Sequential()

#ilk once conv2d ekliyoruz, 32 tane filtre kullaniyoruz ksize yani filtre boyutrumuz 3 e 3 oluyor.
#activation fonksiyonu olarak da relu kullaniyoruz, girdi boyutlarimizi ise yukarda belirttigimiz w h ve 1 olarak belirliyoruz.
model.add(Conv2D(32, kernel_size = (3,3), activation = "relu", input_shape = (width, height, 1)))

#bir tane daha conv layer ekliyoruz, bundan istedigimiz kadar ekleyuebiliriz ve istedigimiz kadar karmasiklastirabiliriz.
#bir oncekinin ciktigi buranin girdisi olacagi icin tekrardan boyutlari falan belirtmemize gerek yok.
#burda karmasikligi artirdik 64 yaptik.
model.add(Conv2D(64, kernel_size = (3,3), activation = "relu"))

#simdi ise pooling yani piksek eklemeyi yapiyoruz.
model.add(MaxPooling2D(pool_size = (2,2)))

#simdi ise seyreltme ekliyoruz.
model.add(Dropout(0.25))

#soimdi ise duzlestirme islemi yapiyopruz.
model.add(Flatten())

#burada ise siniflandirma islemini gerceklestiriyoruz. burad 128 tane noron yapiyoruz, aktivasyon ise relu yapiyoruz.
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.4))

#burasi ise bizim cikti layerimiz. softmax yapiyoruz. softmax fonksiyonunu 2 den fazla cikti varsa kullaniyoruz.
model.add(Dense(3, activation = "softmax"))



# if os.path.exists("./trex_weight.h5"):
#     model.load_weights("trex_weight.h5")
#     print("Weights yuklendi")    



#burda ise modelimizin compile etmek icin gerekli kodlarimizi yaziyoruz.
#loss fonksiyonu bizim en son niahi olarak hatalarimizi hesaplamamizi saglayan fonksiyon. peki bu ne ise yariyor;
#baslangicta hatamiz yani kayiplarimiz cok yuksek cikiyor ve bu kaybimiza gore parametrelerimizi guncelliyoruz,
#geriye dogru turev alma islemi yapiyoruz ve sonucunda da bizim degisimimizi buluyoruz ve bu degisime gore paramtrelerimizi
#guncelliyoruz. bu turev alma islemini de ortaya cikan loss degerimize gore yapiyoruz.
#eger loss cok azsa bu bizim modelimizin iyi egitildigi anlamina geliyor.

#optimizer ise bizim parametrelerimizi optimize ediyor burda gradient descent algoritmasi kullaniliyor.

#metrics ise bizim modelimizin sonuclarinizi yorumlamamiz icin gerekli olan yapidir. bize yuzde olarak basariyi soyler.
model.compile(loss = "categorical_crossentropy", optimizer = "Adam", metrics = ["accuracy"])

#artik training islemine geldik. train x bizim resimleri iceren yapi train y ise bunlarin etiketlerini iceren yapi. 
#epochs ise bizim resimlerimizin toplamda kac kez egitilecegi anlamina geliyor. 35 kere egitim iterasyon gerceklessin diyoruz.
#batch size ise bizim resimlerimizin kac grup halinde itreasyona sokulacagini soyluyoruz.
#yani ilk 64 sonra 64 sonra 64 sonra geri kalanlari egitime sokuyoruz. bu total islem ise bir tane epochs anlamina gelmektedir.
#bunu epochs sayisi kadar tekrarliyor.
model.fit(train_X, train_y, epochs = 35, batch_size = 64)

#sonuc olarak ortaya score train degerimiz cikacak. score un 0. indeksi bize kaybi 1. indeksi ise bize accuracy i dondurur.
#bunu 100 ile carpinca bize egitim dogrulugunun yuzdesini vericektir.
score_train = model.evaluate(train_X, train_y)
print("Eğitim doğruluğu: %",score_train[1]*100)    

#aynisini burda test dogrulugu icin de yapiyoruz.
score_test = model.evaluate(test_X, test_y)
print("Test doğruluğu: %",score_test[1]*100)      
    
 
open("model_new.json","w").write(model.to_json())
model.save_weights("trex_weight_new.h5")   




































































