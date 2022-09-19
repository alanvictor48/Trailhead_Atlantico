import cv2
import mahotas

def write(img, texto, point):
  fonte = cv2.FONT_HERSHEY_SIMPLEX 
  cv2.putText(img, texto, point, fonte, 1, (0, 0, 255), 0, cv2.LINE_AA)

dices = cv2.imread('imgs/dices.png')

gray = cv2.cvtColor(dices, cv2.COLOR_BGR2GRAY)
suave = cv2.bilateralFilter(gray, 11, 100, 100)

T = mahotas.thresholding.otsu(suave)
tmp = suave.copy()
tmp[tmp>T] = 0
tmp[tmp>0] = 255

canny = cv2.Canny(tmp, 70, 150)

objetos, hierarquia = cv2.findContours(canny.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

objetos,_ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

d_img = dices.copy()

for obj in objetos:
  (x, y, w, h) = cv2.boundingRect(obj)
  d = d_img[y-5:y+h+5, x-5:x+w+5]

  gray_d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
  d_suave = cv2.bilateralFilter(gray_d, 11, 100, 100)
  Td = mahotas.thresholding.otsu(d_suave)
  tmpD = d_suave.copy()
  tmpD[tmpD>Td] = 0
  tmpD[tmpD>0] = 255

  bordas = cv2.Canny(tmpD, 70, 150)
  
  objetosD, hier = cv2.findContours(bordas, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

  pips=0
  for h in hier:
    for i in h:
      if i[3]==-1:
        pips+=1

  write(d_img, str(pips-1), (x-5, y-5))
  cv2.drawContours(d, objetosD, -1, (255, 0, 0), 2)
  print('Size:', end=' ')
  print(pips-1)

cv2.imshow('Dices', d_img)
cv2.waitKey(0)