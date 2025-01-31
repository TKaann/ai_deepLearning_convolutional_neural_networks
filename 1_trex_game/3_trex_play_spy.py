from keras.models import model_from_json
import numpy as np
from PIL import Image
import keyboard
import time
from mss import mss


#bunlar bizim trex oyunumuzdaki trexle engelleri gordugumuz yer, modelimiz burayi gorecek ve burdaki
#engelleri gorunce tuslara bakicak bizim algoritmamiz.
mon = {"top":300, "left":770, "width":250, "height":100}
sct = mss()

width = 125
height = 50

#egittigimiz modeli yukluyoruz.
model = model_from_json(open("model.json","r").read())
model.load_weights("trex_weight.h5")

#down = 0, right = 1, up = 2
labels = ["Down", "Right", "Up"]


framerate_time = time.time()
counter = 0
i = 0

#bir komut verdikten sonra 0.4 saniye baska bir komut vermemesini istiyoruz ki o komut genelde yalis oluyor,
#yani havada komut vermemesi icin delay koyuyoruz.
delay = 0.4
key_down_pressed = False

while True:
    #mon bizim belirlemis oldugumuz piksellerdi ve bu pikseller dogrultusunda bir resim aliyorum ve donusturme islemi yapiyoruz.
    img = sct.grab(mon)
    #burada ise numpy array e gore convert ediyoruz ve resize isleminin yapiyoruz.ve normalize ediyoruz.
    im = Image.frombytes("RGB", img.size, img.rgb)
    im2 = np.array(im.convert("L").resize((width, height)))
    im2 = im2 / 255
    
    #bura ise kerasin alabilecegi formata gore convert isllemi yapiyoruz.
    X =np.array([im2])
    X = X.reshape(X.shape[0], width, height, 1)
    #burda ise tahmin islemini yaptirtiyoruz.
    r = model.predict(X)
    
    #burdaki durumu soyle aciklayalim; simdi elimizde 3 adet tus var bunlarin hangisine basilacak olma ihtimalini
    #tahmin ediyoruz. mesella 1 up olsun 2 down 3 right ollsun
    #1 e basma orani 0.3 - 2 ye basma orani 0.6 - 3 e basma orani 0.1 ollsun. bunllarin toplami 1 olacaktir.
    #bizde bunlarin arsindan max olanini yani basilacak olan tuslardan hangisinin tahmin orani daha yuksekse onu seciyoruz.
    result = np.argmax(r)
    
    #bizim icin onemlli olanlalr 0 ve 2 cunku 1 deki right hicbisye yapma demek anllamina gelmektedir.
    #burada ise eger resullt 0 ise yani down ise 
    if result == 0: # down = 0
        
        #burada asagi tusuna bastirtiyoruz.
        keyboard.press(keyboard.KEY_DOWN)
        key_down_pressed = True
    #result 2 ise yani up ise
    elif result == 2:    # up = 2
        
        #burada ise yukari ziplama isllemini yaptirtiyoruz.
        if key_down_pressed:
            keyboard.release(keyboard.KEY_DOWN)
        #burada yukari basmaya baslini kaldigi icin relase edip normale ceviriyoruz.
        time.sleep(delay)
        keyboard.press(keyboard.KEY_UP)
        
        #burada ise 1500. frame e kadar oyunun hizi normal seyrinde giderken 1500 den sonra hizlaniyor
        #hizlandigi icin de basta koydugumuz sleep bizi yavaslatiyor ve bu sleep islemini azaltiyoruz.
        if i < 1500:
            time.sleep(0.3)
        elif 1500 < i and i < 5000:
            time.sleep(0.2)
        else:
            time.sleep(0.17)
        
        #simdi yukarda vakit gecirdikten sonra asagi tusuna bastirtiyoruz, ama eger biraktirmazsak asagi tusuna basili kaliyor
        #bu yuzden release ile asagi tusunu biraktirtiyoruz.
        keyboard.press(keyboard.KEY_DOWN)
        keyboard.release(keyboard.KEY_DOWN)
    
    counter += 1
    
    #burada ise frame gectikce delay vaktini azalltiyoruz en son 0 a esitlenince dellaysiz birakiyoruz.
    if (time.time() - framerate_time) > 1:
        
        counter = 0
        framerate_time = time.time()
        if i <= 1500:
            delay -= 0.003
        else:
            delay -= 0.005
        if delay < 0:
            delay = 0
            
        print("---------------------")
        print("Down: {} \nRight:{} \nUp: {} \n".format(r[0][0],r[0][1],r[0][2]))
        i += 1




































































