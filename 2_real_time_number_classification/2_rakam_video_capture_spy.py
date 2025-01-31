import cv2
import pickle
import numpy as np



#kameramizdan gelen goruntuyu preprocces etmeden alirsak tahmin yapamayacaktir cunku tahmin fonksiyonuna bu ozelllikllerdeki
#verilerin tahminini yapmasi icin egittik. o yuzden tekrardan cevirme islemini yapiyoruz.
def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img /255.0
    
    return img


#kameramizdan capture islemini yapiyoruz.
cap = cv2.VideoCapture(0)
cap.set(3,480)
cap.set(4,480)

#egitmis oldugumuz modeli iceriye yukluyoruz.
pickle_in = open("model_trained_v4.p","rb")
model = pickle.load(pickle_in)



while True:
    
    success, frame = cap.read()
    
    #burada framemizi arraye ceviriyoruz.
    img = np.asarray(frame)
    #tekrardan resize ediyoruz cunku modelimizi egitirken 32 ye 32 yapmistik tekrardan 32 ye 32 yapiyoruz.
    img = cv2.resize(img, (32,32))
    #simdi ise img mizi preprocess e sokuyoruz cunku modeldekiyle ayni olmak zorunda.
    img = preProcess(img)
    
    #ilk bastaki 1: 1 tane resim oldugnu 32 ye 32 boyutunu sonraki 1 ise channellini yani siyah beyaz olldugnu soylluyor.
    img = img.reshape(1,32,32,1)
    
    #predict islemimiz float cikmasin diye int e ceviriyoruz.
    classIndex = int(model.predict_classes(img))
    
    #buranin ciktisi su sekilde olacaktir; 0 ollma olasiligi % 60 dir gibi mesela
    predictions = model.predict(img)
    probVal = np.amax(predictions)
    print(classIndex, probVal)
    
    if probVal > 0.7:
        cv2.putText(frame, str(classIndex)+ "   "+ str(probVal), (50,50),cv2.FONT_HERSHEY_DUPLEX, 1,(0,255,0),1)

    cv2.imshow("Rakam Siniflandirma",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break  































































