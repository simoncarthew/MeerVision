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
        sleep(0.5)
        control.camera.capture("Control/test.jpg")