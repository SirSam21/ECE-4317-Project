import cv2
import numpy as np
from imutils.contours import sort_contours
import imutils
try:
    import docx
except ModuleNotFoundError:
    print('You are not running windows')


def get_lines(image, show=False):
    """
    :return: list of roi parameters for lines
    """
    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if show:
        cv2.imshow('gray', gray)
        cv2.waitKey(0)

    # binary
    ret, thresh = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY_INV)  # 127 changed to 0
    if show:
        cv2.imshow('second', thresh)
        cv2.waitKey(0)

    # dilation
    kernel = np.ones((5, 125), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    if show:
        cv2.imshow('dilated', img_dilation)
        cv2.waitKey(0)

    # find contours
    ctrs = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ctrs = imutils.grab_contours(ctrs)

    # sort contours
    sorted_ctrs = sort_contours(ctrs, method="top-to-bottom")[0]
    l_xs = []
    l_ys = []
    l_ws = []
    l_hs = []

    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y + h, x:x + w]
        l_xs.append(x)
        l_ys.append(y)
        l_ws.append(w)
        l_hs.append(h)
        # show ROI
        if show:
            cv2.imshow('segment no:' + str(i), roi)
            cv2.rectangle(image, (x, y), (x + w, y + h), (90, 0, 255), 2)
            cv2.waitKey(0)
    if show:
        cv2.imshow('marked areas', image)
        cv2.waitKey(0)

    return l_xs, l_ys, l_ws, l_hs


def to_docx(c, filepath):
    strings = ""
    for x in c:
        strings += x

    # Create a new docx at specified file location and write the string on it
    doc = docx.Document()
    doc.add_paragraph(strings)
    doc.save(filepath)

def to_odt(c, filepath):
    print('Performing to_odt...')
    print(f'{c} \n {filepath}')