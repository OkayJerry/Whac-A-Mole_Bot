# https://www.classicgame.com/game/Whack+a+Mole

import cv2
from mss import mss
import numpy as np
from time import sleep
import pyautogui

threshold = 0.8
game_dimensions = (800, 500, 1200, 1100) # x, y, w, h
game_dimensions_mss = {'top': 260, 'left': 410, 'width': 600, 'height': 500}

nose_gray = cv2.imread('images/mole_nose.png', cv2.IMREAD_GRAYSCALE)

h = nose_gray.shape[0]
w = nose_gray.shape[1]

# Allows 3 seconds to open up the correct window / tab
sleep(3)

while True:
    sct = np.array(mss().grab(game_dimensions_mss))
    sct_gray = cv2.cvtColor(sct, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(sct_gray, nose_gray, cv2.TM_CCOEFF_NORMED)

    # Locate matches
    y_loc, x_loc = np.where(result >= threshold)

    # Draw rectangles
    rectangles = []
    for x, y in zip(x_loc, y_loc):
        rectangle_dimensions = (x, y, w, h)
        rectangles.append([int(i) for i in rectangle_dimensions])
        rectangles.append([int(i) for i in rectangle_dimensions])

    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)

    for x, y, _, _ in rectangles:
        cv2.rectangle(sct, (x, y), (x + w, y + h), (0, 0, 255), -1)

    # Show computer's vision
    sct_mini = cv2.resize(sct, (400, 400))
    cv2.imshow('Computer Vision', sct_mini)
    cv2.setWindowProperty('Computer Vision', cv2.WND_PROP_TOPMOST, 1)

    # Interacting with screen
    for x, y, _, _ in rectangles:    
        window_x = (game_dimensions[0] + x) // 2 # compensating for library resolution differences
        window_y = (game_dimensions[1] + y) // 2 # compensating for library resolution differences

        pyautogui.click(window_x, window_y)
        
    # Closes the window on 'q' keypress
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        cv2.destroyAllWindows()
        break 