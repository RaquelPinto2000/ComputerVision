import sys
import logging
import numpy as np
import cv2 as cv


#logging.basicConfig ( level=logging.DEBUG,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Filters")
logger.setLevel(logging.INFO) #vai info ou maior para o logger
logger.propagate = False
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


#handler to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG) 
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

#handler to file
file_handler = logging.FileHandler("LoggerFile.log", mode='a')  # 'a' = append
file_handler.setLevel(logging.INFO) 
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def showImg(img):
    cv.imshow('img', img)

    #fechar a janela de imagem
    try:
        while True:
            cv.waitKey(50)
    except KeyboardInterrupt:
        print("Encerrado pelo Ctrl+C")
        cv.destroyAllWindows()

def gaussianFilter(img, sigmaX, sigmaY=None, ksize=(0,0), borderType = cv.BORDER_DEFAULT):
    analyzeableImage = cv.imread(img)
    if sigmaY is None:
        sigmaY=sigmaX
    logger.info("Init Gaussian Blur filter with arguments: image = " + str(img) + ", sigmaX = " + str(sigmaX) + 
                ", sigmaY = " + str(sigmaY) + ", ksize = " + str(ksize) + ", borderType = " + str(borderType))
    imageEnd= cv.GaussianBlur(analyzeableImage,ksize,sigmaX,sigmaY,borderType)
    
    if imageEnd is None:
        logger.error("Gaussian Blur filter failed: result is None")
    else:
        logger.info("Gaussian Blur filter success executed")

    showImg (imageEnd)

def cannyFilter():
    print("GF")

def medianFilter():
    print("GF")
def averageFilter():
    print("GF")
def bilateralFilter():
    print("GF")
def sobelOperator():
    print("GF")
def LaplacianOperator():
    print("GF")




#capturar imagem
def iniciar ():
    imageCaptured = sys.argv[1] 
    imgCap = cv.imread(imageCaptured)
    showImg(imgCap)

    #filtro da imagem
    gaussianFilter(img=imageCaptured,sigmaX=5,ksize=(5,5),borderType=0)


if __name__ == "__main__":
    iniciar()
   



'''
logger.debug("Mensagem de debug")      # aparece no terminal
logger.info("Mensagem informativa")    # aparece no terminal e no arquivo
logger.warning("Aviso importante")     # aparece nos dois
logger.error("Erro crítico")           # aparece nos dois
logger.critical("Erro fatal")          # aparece nos dois

'''