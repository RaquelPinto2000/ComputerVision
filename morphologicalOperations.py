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

logger = logging.getLogger("Morphological Operations")
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


def showImg(nameImage,img):
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


def erosion(img, ksize=3, kernel=None, anchor=(-1,-1), iterations =1, borderType =cv.BORDER_CONSTANT, saveImage = False, nameSaveImage = "erodedImage"):
    """
    Applies erosion to an image and optionally saves it.

    Parameters
    ----------
    img : ndarray
        Input image.
    ksize : int, optional
        Size of the kernel (if `kernel` is None). Must be an odd number >= 3. Default is 3.
    kernel : ndarray, optional
        Custom kernel. Overrides `ksize` if provided.
    anchor : tuple of 2 ints, optional
        Anchor position within the kernel. Default is (-1, -1), which means the center.
    iterations : int, optional
        Number of times erosion is applied. Default is 1.
    borderType : int, optional
        Pixel extrapolation method for border. Default is cv2.BORDER_CONSTANT.
    saveImage : bool, optional
        If True, saves the final image.
    nameSaveImage : str, optional
        Filename for the saved image.

    Returns
    -------
    ndarray
        Eroded image.
    """

    logger.info("Init Erosion with arguments: kernel = " + str(kernel) + " anchor = " + str(anchor) + " iterations = " + str(iterations) 
                + " borderType = " + str(borderType) + ", saveImage = " + str(saveImage))
    
    if kernel is None:
        if (ksize < 3) or (ksize % 2 == 0):
            logger.error("Kernel size must be an odd number >= 3")
            exit(0)
        kernel = np.ones((ksize, ksize), np.uint8)
    else:
        # Check if kernel is a matriz numpy
        if not isinstance(kernel, np.ndarray):
            logger.error("Custom kernel must be a numpy array")
            exit(0)
    
    if anchor != (-1, -1):
        ax, ay = anchor
        if not (0 <= ax < ksize) or not (0 <= ay < ksize):
            logger.warning("Invalid anchor " + str(anchor) + " for kernel size " + str(kernel) + " Anchor values must satisfy 0 <= anchor < ksize. "
            "The system used (-1,-1) for anchor default.")
            anchor =(-1,-1)

    erodedImage = cv.erode(img, kernel, anchor=anchor, iterations=iterations, borderType=borderType)
   
    if erodedImage is None:
        logger.error("Erosion image failed: result is None")
        exit(0)
    else:
        logger.info("Erosion success executed")

    if saveImage:
        saveImg(str(nameSaveImage + ".png"), erodedImage)
    return erodedImage 


def dilation(img, ksize=3, kernel=None, anchor=(-1,-1), iterations =1, borderType =cv.BORDER_CONSTANT, saveImage = False, nameSaveImage = "dilatedImage"):
    """
    Applies dilation to an image and optionally saves it.

    Parameters
    ----------
    img : ndarray
        Input image.
    ksize : int, optional
        Size of the kernel (if `kernel` is None). Must be odd and >= 3. Default is 3.
    kernel : ndarray, optional
        Custom kernel. Overrides `ksize` if provided.
    anchor : tuple of 2 ints, optional
        Anchor position within the kernel. Default is (-1, -1), which means the center.
    iterations : int, optional
        Number of times dilation is applied. Default is 1.
    borderType : int, optional
        Pixel extrapolation method for borders. Default is cv2.BORDER_CONSTANT.
    saveImage : bool, optional
        If True, saves the resulting image. Default is False.
    nameSaveImage : str, optional
        Filename for the saved image. Default is "dilatedImage".

    Returns
    -------
    ndarray
        Dilated image.
    """
    logger.info("Init Dilation with arguments: kernel = " + str(kernel) + ", anchor = " + str(anchor) + ", iterations = " + str(iterations) 
                + ", borderType = " + str(borderType) + ", saveImage = " + str(saveImage))
    
    if kernel is None:
        if (ksize < 3) or (ksize % 2 == 0):
            logger.error("Kernel size must be an odd number >= 3")
            exit(0)
        kernel = np.ones((ksize, ksize), np.uint8)
    else:
        # Check if kernel is a matriz numpy
        if not isinstance(kernel, np.ndarray):
            logger.error("Custom kernel must be a numpy array")
            exit(0)
    
    if anchor != (-1, -1):
        ax, ay = anchor
        if not (0 <= ax < ksize) or not (0 <= ay < ksize):
            logger.warning("Invalid anchor " + str(anchor) + " for kernel size " + str(kernel) + " Anchor values must satisfy 0 <= anchor < ksize. "
            "The system used (-1,-1) for anchor default.")
            anchor =(-1,-1)

    dilatedImage = cv.dilate(img, kernel, anchor=anchor, iterations=iterations, borderType=borderType)
   
    if dilatedImage is None:
        logger.error("Dilation image failed: result is None")
        exit(0)
    else:
        logger.info("Dilation success executed")

    if saveImage:
        saveImg(str(nameSaveImage + ".png"), dilatedImage)
    return dilatedImage 


def morphologicalOperations(img, op, ksize =3, kernel = None, iterations=1, borderType=cv.BORDER_CONSTANT, saveImage = False, nameSaveImage = "morphologicalImage"):
    """
    Applies a specified morphological operation to an image and optionally saves it.

    Parameters
    ----------
    img : ndarray
        Input image.
    op : int
        Morphological operation to apply. Should be one of:
        - cv.MORPH_ERODE - Not Applied
        - cv.MORPH_DILATE - Not Applied
        - cv.MORPH_OPEN
        - cv.MORPH_CLOSE
        - cv.MORPH_GRADIENT
        - cv.MORPH_TOPHAT
        - cv.MORPH_BLACKHAT
        - cv.MORPH_HITMISS - Only supported for CV_8UC1 binary images

    ksize : int, optional
        Size of the kernel (if `kernel` is None). Must be odd and >= 3. Default is 3.
    kernel : ndarray, optional
        Custom kernel to use. Overrides `ksize` if provided. Must be a NumPy array.
    iterations : int, optional
        Number of times the operation is applied. Default is 1.
    borderType : int, optional
        Pixel extrapolation method for borders. Default is cv2.BORDER_CONSTANT.
    saveImage : bool, optional
        If True, saves the resulting image. Default is False.
    nameSaveImage : str, optional
        Filename for the saved image (without extension). Default is "morphologicalImage".

    Returns
    -------
    ndarray
        Image after applying the specified morphological operation.

    Notes
    -----
    - If `kernel` is None, a square kernel of ones with size `(ksize, ksize)` will be used.
    - The saved filename will append the integer `op` value to `nameSaveImage` if saving.
    - Example usage:
        >>> morphologicalOperations(img, cv.MORPH_OPEN, ksize=5)
    """
    morph_names = {
        cv.MORPH_ERODE: "Erode",
        cv.MORPH_DILATE: "Dilate",
        cv.MORPH_OPEN: "Open",
        cv.MORPH_CLOSE: "Close",
        cv.MORPH_GRADIENT: "Gradient",
        cv.MORPH_TOPHAT: "TopHat",
        cv.MORPH_BLACKHAT: "BlackHat",
        cv.MORPH_HITMISS: "HitMiss"
    }

    # Obtém o nome da operação ou string vazia se não estiver no dicionário
    nameOperation = morph_names.get(op, "")

    logger.info("Init Morphological Operation with arguments: op = " + str(nameOperation) + ", kernel = " + str(kernel) + ", iterations = " + str(iterations) 
                + ", borderType = " + str(borderType) + ", saveImage = " + str(saveImage))
    if op == cv.MORPH_HITMISS and (img.dtype != 'uint8' or len(img.shape) != 2):
        logger.error("Operation MORPH_HITMISS only with CV_8UC1 binary images")
        exit(0)

    if kernel is None:
        if (ksize < 3) or (ksize % 2 == 0):
            logger.error("Kernel size must be an odd number >= 3")
            exit(0)
        kernel = np.ones((ksize, ksize), np.uint8)
    else:
        # Check if kernel is a matriz numpy
        if not isinstance(kernel, np.ndarray):
            logger.error("Custom kernel must be a numpy array")
            exit(0)
    
    morphologicalImage = cv.morphologyEx(img, op, kernel, iterations=iterations, borderType=borderType)
    
    if morphologicalImage is None:
        logger.error("Morphological operation " + nameOperation + " failed: result is None")
        exit(0)
    else:
        logger.info("Morphological operation " + nameOperation + " success executed")


    if saveImage:
        saveImg(f"{nameSaveImage}{nameOperation}.png", morphologicalImage)

    return morphologicalImage


#capturar imagem
def iniciar ():
    imageCaptured = sys.argv[1] 
    imgCap = cv.imread(imageCaptured)
    showImg("init image", imgCap)

    #filtro da imagem
    erosionImage = erosion(img=imgCap, ksize=3, saveImage=True)
    showImg ("erosionImage", erosionImage)
    dilationImage = dilation(img=imgCap, ksize=3, saveImage=True)
    showImg ("dilationImage", dilationImage)
    morphologicalImage = morphologicalOperations(img=imgCap, op=cv.MORPH_OPEN, ksize=3, saveImage=True)
    showImg ("morphologicalImageOpen", morphologicalImage)
    morphologicalImage = morphologicalOperations(img=imgCap, op=cv.MORPH_CLOSE, ksize=3, saveImage=True)
    showImg ("morphologicalImageClose", morphologicalImage)
    morphologicalImage = morphologicalOperations(img=imgCap, op=cv.MORPH_GRADIENT, ksize=3, saveImage=True)
    showImg ("morphologicalImageGradient", morphologicalImage)
    morphologicalImage = morphologicalOperations(img=imgCap, op=cv.MORPH_TOPHAT, ksize=3, saveImage=True)
    showImg ("morphologicalImageTopHat", morphologicalImage)
    morphologicalImage = morphologicalOperations(img=imgCap, op=cv.MORPH_BLACKHAT, ksize=3, saveImage=True)
    showImg ("morphologicalImageBlackHat", morphologicalImage) 
    imgGray = cv.cvtColor(imgCap, cv.COLOR_BGR2GRAY)
    morphologicalImage = morphologicalOperations(img=imgGray, op=cv.MORPH_HITMISS, ksize=3, saveImage=True)
    showImg ("morphologicalImageHitMiss", morphologicalImage)

if __name__ == "__main__":
    iniciar()
  