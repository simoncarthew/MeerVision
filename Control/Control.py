from time import sleep
import os
import sys
from luma.oled.device import ssd1306
from luma.core.interface.serial import i2c
from PIL import Image, ImageDraw, ImageFont

sys.path.append("Control")
from Camera import Camera
from LCD import LCD

class Control:
    def __init__(self):
        self.camera = Camera()
        self.lcd = LCD()

if __name__ == "__main__":
    control = Control()
    while True:
        control.lcd.display_centered_text("YELLOW TEXT", "BLUE TEXT")
        sleep(2)
        control.lcd.display_scroll_wheel("NUMBER SCROLL",10)
        sleep(2)
        control.lcd.display_scroll_wheel("TEXT SCROLL","ITEM 1")
        sleep(2)
    # while True:
    #     sleep(0.5)
    #     control.camera.capture("Control/test.jpg")