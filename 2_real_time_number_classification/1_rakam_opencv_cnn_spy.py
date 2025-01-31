import numpy as np
import cv2
import os 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle




#suanda datamizin icinde zoomlu resimler yok o yuzden image generetor yardimi ile zoom in & out yaparak zenginlestiricez.

path = "myData"

myList = os.listdir(path)
noOfClasses = len(myList)

print("Label(sınıf) sayısı: ",noOfClasses)


images = []
classNo = []

#burada klasorun icindeki her dosyaya ve dosyalarin icindeki her resme ulasiyoruz.
for i in range(noOfClasses):
    myImageList = os.listdir(path + "\\"+str(i))
    for j in myImageList:
        img = cv2.imread(path + "\\" + str(i) + "\\" + j)
        #resimleri resize ederek 32 ye 32 yapiyoruz cunku bizim cnn egitimimizin girdileri 32 ye 32 olucak.
        img = cv2.resize(img, (32,32))
        images.append(img)
        classNo.append(i)
        
print(len(images))
print(len(classNo))

#bundan sonra imagesleri np arraye ceviriyortuz cunku bundna sonra np arrayle islemler yapilabiliyor.
images = np.array(images)
classNo = np.array(classNo)

print(images.shape)
print(classNo.shape)


#veriyi ayırma islemi, ilk once train ve test olarak 2 ye ayiriyoruz sonrasinda ise tekrardan
#train veri setimizi 2 ye ayiriyoruz ve train ve validation elde edicez.

x_train, x_test, y_train, y_test = train_test_split(images, classNo, test_size = 0.5, random_state = 42)
x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size = 0.2, random_state = 42)

#bunun mantigi ise sudur, verimizi train ve test olarak 2 ye boluyoruz, sonrasinda bu testi sakliyoruz. egitimin sonuna kadar
#gormeyecek. veriyi egitirken validation yapmamiz gerekecek yani dogrulama yapmamiz gerekecek. veriyi x tarin ve 
#x validaytion kullanarak egiticez en sonda verimiz hazir dedigimiz anda ise test verisetini kullanarak 
#dogrulama islemlerimizi gerceklestirecegiz.

print(images.shape)
print(x_train.shape)
print(x_test.shape)
print(x_validation.shape)




#burada elimizdeki verileri gorsellestiriyoruz.

# fig, axes = plt.subplots(3,1,figsize=(7,7))
# fig.subplots_adjust(hspace = 0.5)
# sns.countplot(y_train, ax = axes[0])
# axes[0].set_title("y_train")

# sns.countplot(y_test, ax = axes[1])
# axes[1].set_title("y_test")

# sns.countplot(y_validation, ax = axes[2])
# axes[2].set_title("y_validation")



#simdi ise preprocess islemlerimizi yapiyoruz
def preProcess(img):
    #resmimizi gray scale a ceviriyoruz.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #histogramimizi 0 255 arasinda genislettim ve sonrasinda
    img = cv2.equalizeHist(img)
    #0 255 arasi genislettigimiz histogramimizi tekrardan normalize ederek 0 ve 1 arasina getirdik.
    img = img /255
    
    return img


#gray scale e cevirdigimiz resimlere bakiyoruz, veri gorsellestirmesini yapiyoruz. boyutunu da buyuttuk gormek icin tabi.
# idx = 311
# img = preProcess(x_train[idx])
# img = cv2.resize(img,(300,300))
# cv2.imshow("Preprocess ",img)



#simdi bu preProcces isllemini tum verimize uygulayalim.
#map fonksiyonu suna yariyor; 2 tane parametre aliyor ilki bir fonksiyon ikincisi bir lliste. ilk parametredeki fonksiyonu
#2. parametredeki llisteye uygulamamiza yariyor.
x_train = np.array(list(map(preProcess, x_train)))
x_test = np.array(list(map(preProcess, x_test)))
x_validation = np.array(list(map(preProcess, x_validation)))

#burdaki -1 x trainin boyutuna gore kendisini ayarlamasini gerektigi anllamina geliyor, geri kalanlari 32 32 1 yaptiriyoruz.
#yani ordaki -1 train listemizde kac eleman oldugunun anlamina geiyor.
x_train = x_train.reshape(-1,32,32,1)
print(x_train.shape)
x_test = x_test.reshape(-1,32,32,1)
x_validation = x_validation.reshape(-1,32,32,1)



#data generate isllemi
#width_shift_range = 0.1, sudur ki; 0.1 oraninda shift ediyor yani kaydir anlamina geliyor.
#height de yukseklikte kaydrma anlamina geliyor.
#zoom range ise adi ustunde yakinlalstirma skallasi, bunun daha da artmasinda fayda var. bu suan yetersiz gelecek.
#rotation range ise dondurme isllemi onu da 10 yapiyoruz.
dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.1,
                             rotation_range = 10)

dataGen.fit(x_train)

#burada da train ve vallidation datallarimizi one hot encoding hale getiriyoruz.
y_train = to_categorical(y_train, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)
y_validation = to_categorical(y_validation, noOfClasses)

#modeimizi insa edelim, illk once sequentiall ille katmanlari olusturuyoruz.
model = Sequential()
#sonrasinda conv katmaninda input shape veriyoruz, filtreyi 8 yapiyoruz ksize imizi da 5 e 5 yapiyoruz padding piksell ekleme.
model.add(Conv2D(input_shape = (32,32,1), filters = 8, kernel_size = (5,5), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

#bir sonraki katmanda ise filtreyi 16 ksize imizi 3 e 3 yapiyouz.
model.add(Conv2D( filters = 16, kernel_size = (3,3), activation = "relu", padding = "same"))
model.add(MaxPooling2D(pool_size = (2,2)))

#yenio veri uretigimiz icin owerfittingi engellemek icin dropout ekliyoruz.
model.add(Dropout(0.2))
#flatten ile duzestirme yapiyoruz.
model.add(Flatten())
#dense ile hidden layeriomizi ekliyoruz., 256 hucre verdik.
model.add(Dense(units=256, activation = "relu" ))
#1 drop out daha attik.
model.add(Dropout(0.2))
#cikti degerleri.
model.add(Dense(units=noOfClasses, activation = "softmax" ))

#onceki derste anlattik.
model.compile(loss = "categorical_crossentropy", optimizer=("Adam"), metrics = ["accuracy"])

batch_size = 250

#simdi egitim yapiyoruz. hist ile gorsellestirme yapicaz o yuzden histe atadik.
hist = model.fit_generator(dataGen.flow(x_train, y_train, batch_size = batch_size), 
                                        validation_data = (x_validation, y_validation),
                                        epochs = 15,steps_per_epoch = x_train.shape[0]//batch_size, shuffle = 1)

#modelimizi depolamasi icin picklle ekledik.
pickle_out = open("model_trained_new.p","wb")
pickle.dump(model, pickle_out)
pickle_out.close()



# %% degerlendirme
#simdi degerlendirme asamasini yapiyoruz gorselllestirme asamasi.

hist.history.keys()

#burada egitim ve validation degerlerinin basari ve basarisizlik durumlarina bakiyoruz.
plt.figure()
plt.plot(hist.history["loss"], label = "Eğitim Loss")
plt.plot(hist.history["val_loss"], label = "Val Loss")
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history["accuracy"], label = "Eğitim accuracy")
plt.plot(hist.history["val_accuracy"], label = "Val accuracy")
plt.legend()
plt.show()


score = model.evaluate(x_test, y_test, verbose = 1)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])


#burada x validationu kullanarak bir tahmin yaptirtiyoruz ve bunun sonucun da da karsimiza cikarn olasillliksal degerlleri
#maksimize eden seyi bulup indeksini cekiyoruz ve bu bizim tahminimizdir diyoruz y_pred_class a atayarak.
#bir de gercek degerine bakiyoruz,. bu da y_pred in aslinda ne oldugu baklallim y_true ille ayni mi?
y_pred = model.predict(x_validation)
y_pred_class = np.argmax(y_pred, axis = 1)
Y_true = np.argmax(y_validation, axis = 1)

#burda da true  ile tahminimizi karsilastiriyoruz.
cm = confusion_matrix(Y_true, y_pred_class)
f, ax = plt.subplots(figsize=(8,8))
sns.heatmap(cm, annot = True, linewidths = 0.01, cmap = "Greens", linecolor = "gray", fmt = ".1f", ax=ax)
plt.xlabel("predicted")
plt.ylabel("true")
plt.title("cm")
plt.show()



























































