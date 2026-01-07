import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

def load_image(path):
    """
    טוענת תמונה ומחזירה אותה כמערך numpy.
    הוראה מס' 1 [cite: 2]
    """
    # קריאת התמונה והחזרה כמערך
    img = plt.imread(path)
    return np.array(img)

def edge_detection(img_array):
    """
    מבצעת זיהוי קצוות על התמונה.
    הוראה מס' 2 [cite: 3]
    """
    # 1. המרה לאפור (Grayscale) על ידי מיצוע 3 הערוצים
    # הוראות 
    # שים לב: אנחנו משתמשים ב-axis=2 כדי למצע את ערוצי הצבע (RGB)
    gray_img = np.mean(img_array, axis=2)

    # 2. הגדרת הפילטרים (Kernels)
    # הוראות [cite: 8, 9, 10]
    kernelY = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    
    kernelX = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])

    # 3. ביצוע קונבולוציה (Convolution)
    # הוראות [cite: 12, 13, 14, 15]
    # mode='same' -> שומר על גודל התמונה המקורי
    # boundary='fill', fillvalue=0 -> ריפוד באפסים (padding=0)
    edgeY = convolve2d(gray_img, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_img, kernelX, mode='same', boundary='fill', fillvalue=0)

    # 4. חישוב המגניטודה (העוצמה הכוללת)
    # הוראה [cite: 18, 19]
    edgeMAG = np.sqrt(np.square(edgeX) + np.square(edgeY))
    
    return edgeMAG
