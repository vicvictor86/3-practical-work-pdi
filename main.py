import os
from venv import create
import numpy as np
import cv2

def segmentImage(frame):
    #Conversão de BGR para HSV
    #Conversão para HSV pois facilita a segmentação por cor já que existe o canal isolado para a cor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #Definção de valores de limiarização para uma certa cor que será igual próxima da cor da vassoura
    color1 = (60, 60, 55)
    color2 = (105, 255, 255)
    lowThreshold = color1
    highThreshold = color2

    #Limiarização da imagem
    resultImage = cv2.inRange(img, lowThreshold, highThreshold)

    return resultImage

def reduceNoise(image):
    #Definição de um elemento estruturante para remoção de ruídos
    #De acordo com a documentação do cv2 o elemento estruturante MORPH_CROSS é um elemento estruturante de cruz
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

    #Para reduzir os ruidos se faz a abertura pois a erosão irá remover os ruidos
    #Enquanto a dilatação irá retornar o objeto ao seu tamanho original
    #Erosão da imagem
    erodedImage = cv2.erode(image, kernel, iterations = 1)

    #Dilatação da imagem
    dilatedImage = cv2.dilate(erodedImage, kernel, iterations = 1)

    return dilatedImage

def getEachColorChannel(image):
    blueChannel = image[:,:,0]
    greenChannel = image[:,:,1]
    redChannel = image[:,:,2]
    
    return blueChannel, greenChannel, redChannel

def thickenSaber(image):
    structuringElement = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    
    #Colocar só isso engrossa demais o sabre
    #faz-se necessário erodir a imagem para que o sabre não fique muito grosso
    img = cv2.dilate(image, structuringElement, iterations = 2)

    # img = cv2.erode(image, structuringElement, iterations = 1)
    
    return img

def glowSaber(image):
    for i in range(5): 
        blurryImage = cv2.GaussianBlur(image, (37,  37), 0)
        thickenBlurryImage = thickenSaber(blurryImage)
        finalImage[:,:,0] = cv2.add(blueChannel, cv2.subtract(thickenBlurryImage, 40))  # B = 255 - 40 = 215
        finalImage[:,:,1] = cv2.add(greenChannel, cv2.subtract(thickenBlurryImage, 150)) # G = 255 - 150 = 105
        finalImage[:,:,2] = cv2.add(redChannel, cv2.subtract(thickenBlurryImage, 80))  # R = 255 - 80 = 175

    return finalImage

def growingSaber(image, imageToCompare, iterations):
    #Definição de um elemento estruturante para remoção de ruídos
    #De acordo com a documentação do cv2 o elemento estruturante MORPH_CROSS é um elemento estruturante de cruz
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

    #Criando uma máscara igual a imagem que se quer compara para utilizar na operação de interseção
    mask = imageToCompare.copy()
    #Dilata a imagem partindo do ponto inicial de surgimento do sabre de luz
    dilatedImage = cv2.dilate(image, structuringElement, iterations = iterations)
    #Faz a interseção com a imagem que tem o conjunto inicial do sabre de luz e o sabre de luz completo
    dilatedImage = cv2.bitwise_and(dilatedImage, imageToCompare, mask)

    return dilatedImage

def searchInitialSaberPoint(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Definção de valores de limiarização para uma certa cor que será igual próxima da cor da vassoura,
    color1 = (170, 32, 120)
    color2 = (359, 255, 255)
    lowThreshold = color1
    highThreshold = color2

    #Limiarização da imagem
    resultImage = cv2.inRange(img, lowThreshold, highThreshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    resultImage = cv2.erode(resultImage, kernel, iterations = 1)
    resultImage = cv2.dilate(resultImage, kernel, iterations = 1)

    return resultImage

def initialPointWithCompleteSaber(initialPoint, saberImage):
    mask = saberImage.copy()
    newImage = cv2.bitwise_or(initialPoint, saberImage, mask)

    return newImage

video = cv2.VideoCapture('saberVideo.mp4')

frames = []
count = 0

def createDirectories():
    if not os.path.exists('segmentedImages'):
        os.mkdir("segmentedImages")

    if not os.path.exists('reduceNoiseImages'):
        os.mkdir("reduceNoiseImages")

    if not os.path.exists('withoutSaberImages'):
        os.mkdir("withoutSaberImages")

    if not os.path.exists('finalResultImages'):
        os.mkdir("finalResultImages")

    if not os.path.exists('everyChannelImages'):
        os.mkdir("everyChannelImages")

    if not os.path.exists('thickenSaberImages'):
        os.mkdir("thickenSaberImages")

    if not os.path.exists('finalResultImages2'):
        os.mkdir("finalResultImages2")

    if not os.path.exists('growingSaberImages'):
        os.mkdir("growingSaberImages")

    if not os.path.exists('frameImages'):
        os.mkdir("frameImages")

    if not os.path.exists('initialPointImages'):
        os.mkdir("initialPointImages")

    if not os.path.exists('initialPointWithCompleteSaberImages'):
        os.mkdir("initialPointWithCompleteSaberImages")

createDirectories()

# frame_width = int(video.get(3))
# frame_height = int(video.get(4))

# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

while True:
    ret, frame = video.read()
    if ret:
        
        frame = cv2.flip(frame, 0)
        cv2.imwrite(os.path.join("frameImages", f"frame{count}.jpg"), frame)

        segmentInitialPoint = searchInitialSaberPoint(frame)
        cv2.imwrite(os.path.join("initialPointImages", f"segmented{count}.jpg"), segmentInitialPoint)

        segmentedImage = segmentImage(frame)
        cv2.imwrite(os.path.join("segmentedImages", f"frame{count}.jpg"), segmentedImage) 

        segmentInitialPointWithCompleteSaber = initialPointWithCompleteSaber(segmentInitialPoint, segmentedImage)
        cv2.imwrite(os.path.join("initialPointWithCompleteSaberImages", f"segmented{count}.jpg"), segmentInitialPointWithCompleteSaber)

        growingSaberImages = growingSaber(segmentInitialPoint, segmentInitialPointWithCompleteSaber, count)
        cv2.imwrite(os.path.join("growingSaberImages", f"frame{count}.jpg"), growingSaberImages)

        # reducedNoiseImage = reduceNoise(segmentedImage)
        # cv2.imwrite(os.path.join("reduceNoiseImages", f"frame{count}.jpg"), reducedNoiseImage)

        # imageWithoutSaber = removeSaber(segmentedImage)
        # cv2.imwrite(os.path.join("withoutSaberImages", f"frame{count}.jpg"), imageWithoutSaber)

        imageWithTchikenSaber = thickenSaber(growingSaberImages)
        cv2.imwrite(os.path.join("thickenSaberImages", f"frame{count}.jpg"), imageWithTchikenSaber)

        blueChannel, greenChannel, redChannel = getEachColorChannel(frame)
        cv2.imwrite(os.path.join("everyChannelImages", f"frame{count}_redChannel.jpg"), redChannel)

        finalImage = frame

        #Aumenta a intensidade de cada canal de cor de acordo com o resultado da imagem
        #Nesse caso temos uma imagem com fundo preto e apenas as regiões da vassoura branca
        #Então o que se faz é aumentar a intensidade de cada canal de cor de acordo com a intensidade da imagem
        #Porém até agora, como é possível ver nos frames, ainda é possível ver a vassoura e o efeito do sabre ainda está fraco
        finalImage[:,:,0] = cv2.add(blueChannel, imageWithTchikenSaber)
        finalImage[:,:,1] = cv2.add(greenChannel, imageWithTchikenSaber)
        finalImage[:,:,2] = cv2.add(redChannel, imageWithTchikenSaber)

        cv2.imwrite(os.path.join("finalResultImages", f"frame{count}.jpg"), finalImage)

        # saberGrowing = growingSaber(imageWithoutSaber, reducedNoiseImage)
        # cv2.imwrite(os.path.join("growingSaberImages", f"frame{count}.jpg"), saberGrowing)

        finalImage = glowSaber(imageWithTchikenSaber)

        cv2.imwrite(os.path.join("finalResultImages2", f"frame{count}.jpg"), finalImage)

        if count > 10:
            break

        count += 1
    else:
        print("Fim do vídeo")
        break