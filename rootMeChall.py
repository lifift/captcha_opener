import requests
import base64
import numpy as np
import cv2
import os
from tensorflow.keras import models, layers, utils, backend as K, datasets
from PIL import Image as I
from PIL import ImageOps
import matplotlib.pyplot as plt

#PREPARATION DU MODEL

tmpchars="abcdefghijkmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ23456789"
chars={}
k=1
for char in list(tmpchars):
    code = []
    for i in range (len(tmpchars)):
        if i+1 == k:
            code.append(1.0)
        else :
            code.append(0.0)
    k+=1
    chars[char]=np.asarray(code)
    

model = models.load_model('current_model')
a=1
#REQUETE HTTP 
rep ="Failed, Try again"
while len(rep.split("Failed, Try again"))==2:
    x= requests.get("http://challenge01.root-me.org/programmation/ch8/")
    img=x.text.split("base64,")[1].split('" /><br><br><form action=""')[0]
    #print(img)
    img=base64.b64decode(img)
    jpg_as_np = np.frombuffer(img, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    COKIE = x.cookies

    # COKIE['_ga'] ='GA1.1.774358813.1651247133'
    # COKIE['_ga_SRYSKX09J7']='GS1.1.1653496779.25.1.1653499243.0'



    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),3)
    #cv2.imshow('gray', cv2.resize(img,(1000,300)))
    gray = cv2.resize(gray,(3000,400))
    (thresh, gray) = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7,7), np.uint8)
    kernel2 = np.ones((7,7), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.medianBlur(gray,13)
    gray= cv2.resize(gray,(250,50))
    #cv2.imshow('thresh', cv2.resize(gray,(1000,300)))


    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)


    ctrs, hier = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=(lambda ctr: cv2.boundingRect(ctr)[0]))

    #print("Number of contours:" + str(len(ctrs)))
    iname=1
    letterlist=[]
    poslist=[]
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        roi = img[y:y + h, x:x + w]
        area = w*h
        print(area)
        if 70 < area < 500:
            #rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            dim =(28,28)
            resizedim = cv2.resize(roi,dim)
            cv2.imwrite('lettres_tmp/'+list(tmpchars)[iname]+'.png',resizedim)
            letterlist.append(resizedim)
            poslist.append((x,y,w,h))
            #cv2.imshow('thresh', resizedim)
            #cv2.waitKey(0)
            iname+=1
            
    ### TOTRY
    """
    for k in range(len(poslist)) :
        for pos in poslist :
            if (poslist[k][0]>pos[0]) and (poslist[k][0]<pos[0]+pos[2]) and (poslist[k][1]>pos[1]) and (poslist[k][1]<pos[1]+pos[3]) :
                del letterlist[k]
                break
                
    iname=1           
    for im in letterlist :
        if im is not None:
            cv2.imwrite('lettres_tmp/'+list(tmpchars)[iname]+'.png',resizedim)
            iname+=1
            
    """
    ### TOTRY


    resultat_final=""
    liste_images = os.listdir('lettres_tmp/')
    #print(liste_images)
    for name in liste_images:
        with I.open('lettres_tmp/'+name,'r') as im:
            imG=1-((np.asarray(ImageOps.grayscale(im)))/255.0)
            res=model.predict(imG.reshape(1,28,28))
            mostP=(0,0)
            for x in range(len(res[0])):
                if res[0][x]>mostP[1]:
                    mostP=(x,res[0][x])
            print("le plus probable est le charactère : "+ list(tmpchars)[mostP[0]])
            print("La proba est : "+str(mostP[1]))
            resultat_final +=list(tmpchars)[mostP[0]]
            #plt.imshow(imG,cmap='gray')
            #plt.show()

    print(resultat_final)

    data = 'cametu='+resultat_final
    # proxies = {
    #    'http': '127.0.0.1:8080',
    #    'https': '',
    # }
    headers= {
        "Content-Type": "application/x-www-form-urlencoded",
        "Origin": "http://challenge01.root-me.org",
        "Connection": "close",
        "Referer": "http://challenge01.root-me.org/programmation/ch8/",
        "Accept-Language": "fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3",
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0"
    }
    response = requests.post("http://challenge01.root-me.org/programmation/ch8/",headers=headers,data=data,cookies=COKIE)

    rep = response.text.split("<br></p><br/><img")[0]
    print(a)
    a+=1
print(rep)
