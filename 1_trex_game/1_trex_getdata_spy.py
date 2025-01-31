#pip install keyboard

#pip install mss


import keyboard
import uuid
import time
from PIL import Image
from mss import mss


#bu pikseller bizim oyunumuzun engellerinin dinazora yaklastigi yerin kordinatlari, derin ogrenme algoritmamiz bu belirledigimiz
#kordinatlara bakip ona gore ziplama ya da egilme kararini verecektir.
mon = {"top":300, "left":770, "width":250, "height":100}

#mss kutuphanesi bizim bu pikseller dogrultusunda ekrandan ilgil alani ROI yi kesip frame haline donusturecek olan kutuphanemiz.
sct = mss()

i = 0

def record_screen(record_id, key):
    global i
    
    i += 1
    #burdaki key dedigimiz  sey klavyedeki bastigimiz tus olacak i ise kac kez klavyeye bastigimiz oluyor.
    print("{}: {}".format(key, i))
    #burada ise img yi az onceki verdigimiz kordinatlar dogrultusunda aliyoruz.
    img = sct.grab(mon)
    #simdi burada ise ekran goruntulerimizi kaydediyoruz, yani burada data topluyoruz suanda, ne zaman ve hangi engel karsisinda
    #hangi hareketleri yapiyoruz onlar kaydediyoruz. hem ekranda ne oldugunu hem de hangi hareketi yaptigimizi.
    im = Image.frombytes("RGB", img.size, img.rgb)
    im.save("./img/{}_{}_{}.png".format(key, record_id, i))
    
is_exit = False

#burada ise veri toplayamayi birakmak istedigimiz zaman esc tusuna bastigimizda kodumuzdan cikisini gerceklestiren 
#kodumuzu yaziyoruz.
def exit():
    global is_exit
    is_exit = True
    
keyboard.add_hotkey("esc", exit)

record_id = uuid.uuid4()

while True:
    #eger dogru iise cikis islemini gerceklestiriyoruz break yapiyoruz.
    if is_exit: break

    try:
        #burada eger yukari tusuna basarsak record screeni getiricek ve key olarak da up yazicak.
        if keyboard.is_pressed(keyboard.KEY_UP):
            record_screen(record_id, "up")
            #sleep koymamizin sebebi ise verdigimiz komutlarin cakismamasi icin.
            time.sleep(0.1)
            #ayni sekilde burada da asagiya basarsak ayni islemi yapicak.
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            record_screen(record_id, "down")
            time.sleep(0.1)
            #burada da saga basma islemini yaptik neden buna ihtiyacimiz var;
            #cunku siniflandirma yaparken bu ya asagiya egilecek ya da yukari cikicak ziplamasi gereken yerlerde ziplamasi
            #istedigimiz bir durum ama ziplamamasi gereken yerlede ziplarsa bir sonraki hamle icin hatali bir pozsisyonda 
            #kalabilir, bu nedenle ziplamamasi ya da egilmemesi gereken yerderde sag tusuna basicaz ki algoritmamiz aslinda
            #sag tusunun hicbirsey yapmamasi gerektigini zaten ekranin normal bir bicimde akmasi gerektigini ogrensin diye.
        elif keyboard.is_pressed("right"):
            record_screen(record_id, "right")
            time.sleep(0.1)
    #buraya try blogu eklememizin sebebi ise olasi bir hata alirsak dahi kodumuzun durmasi yerine devam etmesini istiyoruz.
    except RuntimeError: continue




























































