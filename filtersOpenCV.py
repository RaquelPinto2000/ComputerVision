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



def showImg(nameImage,img):
    cv.imshow(str(nameImage), img)

    #fechar a janela de imagem
    try:
        while True:
            cv.waitKey(50)
    except KeyboardInterrupt:
        print("Encerrado pelo Ctrl+C")
        cv.destroyAllWindows()

#todo - guardar imagem após filtros numa pasta

def gaussianFilter(img, sigmaX, sigmaY=None, ksize=(0,0), borderType=cv.BORDER_DEFAULT):
    if sigmaY is None:
        sigmaY=sigmaX
    logger.info("Init Gaussian Blur filter with arguments: sigmaX = " + str(sigmaX) + 
                ", sigmaY = " + str(sigmaY) + ", ksize = " + str(ksize) + ", borderType = " + str(borderType))
    gaussianImage= cv.GaussianBlur(img,ksize,sigmaX,sigmaY,borderType)
    
    if gaussianImage is None:
        logger.error("Gaussian Blur filter failed: result is None")
        exit(0)
    else:
        logger.info("Gaussian Blur filter success executed")

    return gaussianImage
   

def cannyFilter(img, thresholdInf, thresholdSup, edgeImage=None, aptureSize=3, L2gradient=False):
    logger.info("Init Canny Edge Detection with arguments: thresholdInf = " + str(thresholdInf) + 
                ", thresholdSup = " + str(thresholdSup) + ", aptureSize = " + str(aptureSize) + ", L2gradient = " + str(L2gradient))
    
    edgesImage = cv.Canny(img, thresholdInf, thresholdSup, edgeImage, aptureSize, L2gradient)
    
    if edgesImage is None:
        logger.error("Canny Edge Detection failed: result is None")
        exit(0)
    else:
        logger.info("Canny Edge Detection success executed")
    return edgesImage


def medianFilter(img, ksize=3):
    logger.info("Init Median Filter with arguments: kernel size = " + str(ksize))
  
    if ((ksize < 3) or ((ksize % 2) == 0)):
        logger.error("The kernel size number must be an odd number greater than or equal to 3")
        medianImage=None
    else:
        medianImage = cv.medianBlur(img, ksize)
    
    if medianImage is None:
        logger.error("Median Filter failed: result is None")
        exit(0)
    else:
        logger.info("Median Filter success executed")
   
    return medianImage


def averageFilter(img, ksize=3, anchor =(-1,-1), borderType=cv.BORDER_DEFAULT, averageImage = None):
    logger.info("Init Average Filter with arguments: kernel size = " + str(ksize) + " anchor = " + str(anchor) + " borderType = " + str(borderType))
    if ((ksize < 3) or ((ksize % 2) == 0)):
        logger.error("The kernel size number must be an odd number greater than or equal to 3")
        averagesImage=None
    else:
        ksizeTuple = (ksize,ksize)

    if anchor != (-1, -1):
        ax, ay = anchor
        if not (0 <= ax < ksize) or not (0 <= ay < ksize):
            logger.warning("Invalid anchor " + str(anchor) + " for kernel size " + str(ksizeTuple) + " Anchor values must satisfy 0 <= anchor < ksize. "
            "The system used (-1,-1) for anchor default.")
            anchor =(-1,-1)
        
    averagesImage = cv.blur(img, ksizeTuple, averageImage, anchor, borderType)
    if averagesImage is None:
        logger.error("Average Filter failed: result is None")
        exit(0)
    else:
        logger.info("Average Filter success executed")
   
    return averagesImage
    #todo - ver boxfilter function

def bilateralFilter(img, sigmaColor, sigmaSpace, ksize=3, borderType=cv.BORDER_DEFAULT):
    logger.info("Init Bilateral Filter with arguments: kernel size = " + str(ksize) + " sigmaColor = " + str(sigmaColor) + " sigmaSpace = " + str(sigmaSpace) + " borderType = " + str(borderType))
    if ((ksize < 3) or ((ksize % 2) == 0)):
        logger.error("The kernel size number must be an odd number greater than or equal to 3")
        bilateralImage=None
    else:
        bilateralImage = cv.bilateralFilter(img, ksize, sigmaColor, sigmaSpace, borderType)
        
    if bilateralImage is None:
        logger.error("Bilateral Filter failed: result is None")
        exit(0)
    else:
        logger.info("Bilateral Filter success executed")
    return bilateralImage
 

def sobelOperator(img, ddepth, dx, dy, ksize=3, scale=1, delta=None, borderType = cv.BORDER_DEFAULT):
    logger.info("Init Sobel Operator with arguments: ddepth = " + str(ddepth) + " dx = " + str(dx) + " dy = " + str(dy) + " ksize = " + str(ksize) + " scale = " + str(scale) + " delta = " + str(delta) + " borderType = " + str(borderType))
    
    if dx not in (0, 1) or dy not in (0, 1):
        logger.error("The dx and dy only can be 1 or 0")
        exit(0)

    if ((ksize < 3) or ((ksize % 2) == 0)):
        logger.error("The kernel size number must be an odd number greater than or equal to 3")
        sobelImage=None
    else:
        sobelImage = cv.Sobel(img, ddepth,  dx, dy, ksize, scale, delta, borderType)      

    if sobelImage is None:
        logger.error("Sobel Operator image failed: result is None")
        exit(0)
    else:
        logger.info("Sobel Operator success executed")
    return sobelImage  



def LaplacianOperator():
    print("GF")




#capturar imagem
def iniciar ():
    imageCaptured = sys.argv[1] 
    imgCap = cv.imread(imageCaptured)
    showImg("init image", imgCap)

    #filtro da imagem
    gaussianImage = gaussianFilter(img=imgCap,sigmaX=5,ksize=(0,0),borderType=0)
    showImg ("gaussianImage", gaussianImage)
    edgeImage = cannyFilter(img=imgCap,thresholdInf=1,thresholdSup=100)
    showImg ("edgeImage",edgeImage)
    medianImage = medianFilter(img=imgCap,ksize=7)
    showImg ("medianImage", medianImage)
    averageImage = averageFilter(img=imgCap,ksize=3,anchor=(3,3))
    showImg ("averageImage",averageImage)
    bilateralImage = bilateralFilter(img=imgCap, ksize=9, sigmaColor=100, sigmaSpace=75)
    showImg ("bilateralImage",bilateralImage)
    sobelOperatorImage = sobelOperator(img=imgCap, ddepth = cv.CV_64F, dx=0, dy=1, ksize=3, scale=1)
    showImg ("sobelOperatorImage",sobelOperatorImage)

if __name__ == "__main__":
    iniciar()
   



'''
logger.debug("Mensagem de debug")      # aparece no terminal
logger.info("Mensagem informativa")    # aparece no terminal e no arquivo
logger.warning("Aviso importante")     # aparece nos dois
logger.error("Erro crítico")           # aparece nos dois
logger.critical("Erro fatal")          # aparece nos dois

'''