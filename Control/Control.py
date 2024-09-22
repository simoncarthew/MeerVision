from time import sleep
import os
import sys
from luma.oled.device import ssd1306
from luma.core.interface.serial import i2c
from PIL import Image, ImageDraw, ImageFont

sys.path.append("Control")
from Camera import Camera
from LCD import LCD
from RTC import RTC
from Buttons import Buttons

class Control:
    def __init__(self):
        self.camera = Camera()
        self.lcd = LCD()
        self.rtc = RTC()
        self.buttons = Buttons(pin_ok=23, pin_up=27, pin_down=22, pin_back=24)

if __name__ == "__main__":
    control = Control()
    try:
        while True:
            print(control.buttons.get_button_states())
            sleep(0.1)  # Add a small delay to prevent CPU overuse
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        control.buttons.cleanup()

    # while True:
    #     sleep(0.5)
    #     control.camera.capture("Control/test.jpg")