def preprocessing(img):
    # conversao da imagem para escala de cinza
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # aplicacao de filtro gaussiano
    imgblur = cv2.GaussianBlur(img_gray, (3,3), 0)
    # aplicacao de filtro laplaciano
    imglap = cv2.Laplacian(imgblur, cv2.CV_16S, ksize=3)
    # conversao para uint8
    imgabs = cv2.convertScaleAbs(imglap)
    # equalizacao de histograma
    imgeq = cv2.equalizeHist(imgabs)
    # filtro de media
    imgblur = cv2.blur(imgeq, (5,5))
    # limiarizacao binaria
    ret,thresh1 = cv2.threshold(imgblur,120,255,cv2.THRESH_BINARY_INV)
    # opening
    kernel = np.ones((13,13),np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)
    # dilatacao
    kernel_dl = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(opening,kernel_dl,iterations = 1)
    
    return dilation