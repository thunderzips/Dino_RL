import pyautogui
import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy
import time

#on 200% zoom on firefox

screenshot = pyautogui.screenshot(region=(300,200, 1920-400, 1080//2-50))
# cv2.imwrite("saved.png", np.array(screenshot))

game_over_img = np.array(copy.copy(screenshot))[250:275, 400:900]
# cv2.imwrite("game_over_img.png",game_over_img)

if np.mean(game_over_img) > 230:
    game_over = False
else:
    game_over = True

print(game_over)

# keyboard.wait("Ctrl")
# keyboard.press_and_release('up')

time.sleep(3)
pyautogui.hotkey('up')