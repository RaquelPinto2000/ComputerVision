import sys
import logging
import numpy as np
import cv2 as cv
import os

'''
logger.debug("Mensagem de debug")      # aparece no terminal
logger.info("Mensagem informativa")    # aparece no terminal e no arquivo
logger.warning("Aviso importante")     # aparece nos dois
logger.error("Erro crítico")           # aparece nos dois
logger.critical("Erro fatal")          # aparece nos dois

'''

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
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
results_path = os.path.join(PROJECT_PATH, "Results")
os.makedirs(results_path, exist_ok=True)


def showImg(nameImage,img): #debug
    cv.imshow(str(nameImage), img)

    #fechar a janela de imagem
    try:
        while True:
            cv.waitKey(50)
    except KeyboardInterrupt:
        print("Encerrado pelo Ctrl+C")
        cv.destroyAllWindows()


def saveImg(nameImage,img):
    cv.imwrite(os.path.join(results_path, nameImage), img)


def gaussianFilter(img, sigmaX, sigmaY=None, ksize=(0,0), borderType=cv.BORDER_DEFAULT, saveImage = False, nameSaveImage = "gaussianImage"):
    """
    Applies Canny edge detection to an image and optionally saves it.

    Parameters
    ----------
    img : ndarray
        Input image.
    thresholdInf : float
        Lower threshold for edge detection.
    thresholdSup : float
        Upper threshold for edge detection.
    edgeImage : ndarray, optional
        Optional output image to store edges. Default is None.
    aptureSize : int, optional
        Aperture size for the Sobel operator. Default is 3.
    L2gradient : bool, optional
        Whether to use a more accurate L2 norm. Default is False.
    saveImage : bool, optional
        If True, saves the resulting image. Default is False.
    nameSaveImage : str, optional
        Filename for the saved image. Default is "cannyImage".

    Returns
    -------
    ndarray
        Edge-detected image.
    """

    if sigmaY is None:
        sigmaY=sigmaX
    logger.info("Init Gaussian Blur filter with arguments: sigmaX = " + str(sigmaX) + 
                ", sigmaY = " + str(sigmaY) + ", ksize = " + str(ksize) + ", borderType = " + str(borderType) + ", saveImage = " + str(saveImage))
    gaussianImage= cv.GaussianBlur(img,ksize,sigmaX,sigmaY,borderType)
    
    if gaussianImage is None:
        logger.error("Gaussian Blur filter failed: result is None")
        exit(0)
    else:
        logger.info("Gaussian Blur filter success executed")

    if saveImage:
        saveImg(str(nameSaveImage + ".png"), gaussianImage)
    return gaussianImage
   

def cannyFilter(img, thresholdInf, thresholdSup, edgeImage=None, aptureSize=3, L2gradient=False, saveImage = False,  nameSaveImage = "cannyImage"):
    """
    Applies Canny edge detection to an image and optionally saves it.

    Parameters
    ----------
    img : ndarray
        Input image.
    thresholdInf : float
        Lower threshold for edge detection.
    thresholdSup : float
        Upper threshold for edge detection.
    edgeImage : ndarray, optional
        Optional output image to store edges. Default is None.
    aptureSize : int, optional
        Aperture size for the Sobel operator. Default is 3.
    L2gradient : bool, optional
        Whether to use a more accurate L2 norm. Default is False.
    saveImage : bool, optional
        If True, saves the resulting image. Default is False.
    nameSaveImage : str, optional
        Filename for the saved image. Default is "cannyImage".

    Returns
    -------
    ndarray
        Edge-detected image.
    """
    
    logger.info("Init Canny Edge Detection with arguments: thresholdInf = " + str(thresholdInf) + 
                ", thresholdSup = " + str(thresholdSup) + ", aptureSize = " + str(aptureSize) + ", L2gradient = " + str(L2gradient) 
                + ", saveImage = " + str(saveImage))
    
    cannyImage = cv.Canny(img, thresholdInf, thresholdSup, edgeImage, aptureSize, L2gradient)
    
    if cannyImage is None:
        logger.error("Canny Edge Detection failed: result is None")
        exit(0)
    else:
        logger.info("Canny Edge Detection success executed")
    
    if saveImage:
        saveImg(str(nameSaveImage + ".png"), cannyImage)
    return cannyImage


def medianFilter(img, ksize=3, saveImage = False, nameSaveImage = "medianImage"):
    """
    Applies median filter to an image and optionally saves it.

    Parameters
    ----------
    img : ndarray
        Input image.
    ksize : int, optional
        Size of the kernel. Must be odd and >= 3. Default is 3.
    saveImage : bool, optional
        If True, saves the resulting image. Default is False.
    nameSaveImage : str, optional
        Filename for the saved image. Default is "medianImage".

    Returns
    -------
    ndarray
        Median filtered image.
    """
    
    logger.info("Init Median Filter with arguments: kernel size = " + str(ksize) + ", saveImage = " + str(saveImage))
  
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
    
    if saveImage:
        saveImg(str(nameSaveImage +".png"), medianImage)
    return medianImage


def averageSumFilter(img, ksize=3, ddepth=-1, normalize=True, imageDst = None, anchor =(-1,-1), borderType=cv.BORDER_DEFAULT, saveImage = False, nameSaveImage = "filteredImage"):
    """
    Applies average (box) filter to an image and optionally saves it.

    Parameters
    ----------
    img : ndarray
        Input image.
    ksize : int, optional
        Size of the kernel. Must be odd and >= 3. Default is 3.
    ddepth : int, optional
        Desired depth of the output image. Default is -1 (same as input).
    normalize : bool, optional
        If True, computes average; if False, computes sum. Default is True.
    anchor : tuple of 2 ints, optional
        Anchor position within the kernel. Default is (-1, -1), which means center.
    borderType : int, optional
        Pixel extrapolation method for borders. Default is cv2.BORDER_DEFAULT.
    saveImage : bool, optional
        If True, saves the resulting image. Default is False.
    nameSaveImage : str, optional
        Filename for the saved image. Default is "averagesImage".

    Returns
    -------
    ndarray
        Filtered image.
    """
    logger.info("Init Average Filter with arguments: kernel size = " + str(ksize) + ", ddepth = " + str(ddepth) + ", normalize = " + str(normalize) +
                ", anchor = " + str(anchor) + ", borderType = " + str(borderType) + ", saveImage = " + str(saveImage))
    if ((ksize < 3) or ((ksize % 2) == 0)):
        logger.error("The kernel size number must be an odd number greater than or equal to 3")
        filteredImage=None
    else:
        ksizeTuple = (ksize, ksize)

    if anchor != (-1, -1):
        ax, ay = anchor
        if not (0 <= ax < ksize) or not (0 <= ay < ksize):
            logger.warning("Invalid anchor " + str(anchor) + " for kernel size " + str(ksizeTuple) + " Anchor values must satisfy 0 <= anchor < ksize. "
            "The system used (-1,-1) for anchor default.")
            anchor =(-1,-1)
        
    filteredImage = cv.boxFilter(img, ddepth, ksizeTuple, imageDst, anchor, normalize, borderType)

    if filteredImage is None:
        logger.error("Average Filter failed: result is None")
        exit(0)
    else:
        logger.info("Average Filter success executed")
   
    if saveImage:
        saveImg(str(nameSaveImage + ".png"), filteredImage)
    return filteredImage


def bilateralFilter(img, sigmaColor, sigmaSpace, ksize=3, borderType=cv.BORDER_DEFAULT, saveImage = False, nameSaveImage = "bilateralImage"):
    """
    Applies bilateral filter to an image and optionally saves it.

    Parameters
    ----------
    img : ndarray
        Input image.
    sigmaColor : float
        Filter sigma in color space.
    sigmaSpace : float
        Filter sigma in coordinate space.
    ksize : int, optional
        Diameter of each pixel neighborhood. Must be odd and >= 3. Default is 3.
    borderType : int, optional
        Pixel extrapolation method for borders. Default is cv2.BORDER_DEFAULT.
    saveImage : bool, optional
        If True, saves the resulting image. Default is False.
    nameSaveImage : str, optional
        Filename for the saved image. Default is "bilateralImage".

    Returns
    -------
    ndarray
        Bilateral filtered image.
    """

    logger.info("Init Bilateral Filter with arguments: kernel size = " + str(ksize) + ", sigmaColor = " + str(sigmaColor) 
                + ", sigmaSpace = " + str(sigmaSpace) + ", borderType = " + str(borderType) + ", saveImage = " + str(saveImage))
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
    
    if saveImage:
        saveImg(str(nameSaveImage + ".png"), bilateralImage)
    return bilateralImage
 

def sobelOperator(img, dx, dy, ddepth=cv.CV_64F, ksize=3, scale=1, delta=None, borderType = cv.BORDER_DEFAULT, saveImage = False, nameSaveImage = "sobelImage"):
    """
    Applies Sobel operator to an image and optionally saves it.

    Parameters
    ----------
    img : ndarray
        Input image.
    dx : int
        Order of derivative in x direction (0 or 1).
    dy : int
        Order of derivative in y direction (0 or 1).
    ddepth : int, optional
        Desired depth of the output image. Default is cv2.CV_64F.
    ksize : int, optional
        Size of the Sobel kernel. Must be odd and >= 3. Default is 3.
    scale : float, optional
        Optional scale factor for the computed derivative. Default is 1.
    delta : float, optional
        Optional value added to the results. Default is None.
    borderType : int, optional
        Pixel extrapolation method for borders. Default is cv2.BORDER_DEFAULT.
    saveImage : bool, optional
        If True, saves the resulting image. Default is False.
    nameSaveImage : str, optional
        Filename for the saved image. Default is "sobelImage".

    Returns
    -------
    ndarray
        Image after applying Sobel operator.
    """
    
    logger.info("Init Sobel Operator with arguments: ddepth = " + str(ddepth) + ", dx = " + str(dx) + ", dy = " + str(dy) + ", ksize = " + str(ksize) 
                + ", scale = " + str(scale) + ", delta = " + str(delta) + ", borderType = " + str(borderType) + ", saveImage = " + str(saveImage))
    
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
    
    if saveImage:
        saveImg(str(nameSaveImage + ".png"), sobelImage)
    return sobelImage  


def laplacianOperator(img, ddepth = cv.CV_64F, ksize=3, scale=1, delta=0, borderType = cv.BORDER_DEFAULT, saveImage = False, nameSaveImage = "laplacianImage"):
    """
    Applies Laplacian operator to an image and optionally saves it.

    Parameters
    ----------
    img : ndarray
        Input image.
    ddepth : int, optional
        Desired depth of the output image. Default is cv2.CV_64F.
    ksize : int, optional
        Kernel size. Must be odd and >= 1. Default is 3.
    scale : float, optional
        Optional scale factor for the computed Laplacian. Default is 1.
    delta : float, optional
        Optional value added to the results. Default is 0.
    borderType : int, optional
        Pixel extrapolation method for borders. Default is cv2.BORDER_DEFAULT.
    saveImage : bool, optional
        If True, saves the resulting image. Default is False.
    nameSaveImage : str, optional
        Filename for the saved image. Default is "laplacianImage".

    Returns
    -------
    ndarray
        Image after applying Laplacian operator.
    """
    
    logger.info("Init Laplacian Operator with arguments: ddepth = " + str(ddepth) + ", ksize = " + str(ksize) + ", scale = " + str(scale) 
                + ", delta = " + str(delta) + ", borderType = " + str(borderType) + ", saveImage = " + str(saveImage))
    if scale < 0:
        logger.error("The scale number must be greater than or equal to 0")
        exit(0)

    if ((ksize < 1) or ((ksize % 2) == 0)):
        logger.error("The kernel size number must be an odd number greater than or equal to 1")
        laplacianImage=None
    else:
        laplacianImage = cv.Laplacian(src=img, ddepth=ddepth,ksize=ksize, scale=scale, delta=delta, borderType=borderType)
    
    if laplacianImage is None:
        logger.error("Laplacian Operator failed: result is None")
        exit(0)
    else:
        logger.info("Laplacian Operator success executed")
    
    if saveImage:
        saveImg(str(nameSaveImage + ".png"), laplacianImage)
    return laplacianImage



#capturar imagem
def iniciar ():
    imageCaptured = sys.argv[1] 
    imgCap = cv.imread(imageCaptured)
    showImg("init image", imgCap)

    #filtro da imagem
    gaussianImage = gaussianFilter(img=imgCap,sigmaX=5,ksize=(0,0),borderType=0, saveImage=True, nameSaveImage="gaussianImage")
    showImg ("gaussianImage", gaussianImage)
    edgeImage = cannyFilter(img=imgCap,thresholdInf=1,thresholdSup=100, saveImage=True, nameSaveImage="cannyImage")
    showImg ("edgeImage",edgeImage)
    medianImage = medianFilter(img=imgCap,ksize=7, saveImage=True, nameSaveImage="medianImage")
    showImg ("medianImage", medianImage)
    filteredImage = averageSumFilter(img=imgCap,ksize=3,anchor=(1,1), saveImage=True, nameSaveImage="filteredImage")
    showImg ("filteredImage",filteredImage)
    bilateralImage = bilateralFilter(img=imgCap, ksize=9, sigmaColor=100, sigmaSpace=75, saveImage=True, nameSaveImage="bilateralImage")
    showImg ("bilateralImage",bilateralImage)
    sobelOperatorImage = sobelOperator(img=imgCap, ddepth = cv.CV_64F, dx=0, dy=1, ksize=3, scale=1, saveImage=True, nameSaveImage="sobelOperatorImage")
    showImg ("sobelOperatorImage",sobelOperatorImage)
    laplacianOperatorImage = laplacianOperator(img=imgCap, ddepth = cv.CV_64F, ksize=3, scale=1, saveImage=True, nameSaveImage="laplacianOperatorImage")
    showImg ("laplacianOperatorImage",laplacianOperatorImage)

if __name__ == "__main__":
    iniciar()
   