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
        
        # over ride the burron actions
        self.buttons.ok_action = self.ok_action
        self.buttons.back_action = self.back_action
        self.buttons.up_action = self.up_action
        self.buttons.down_action = self.down_action

    def ok_action(self):
        pass

    def back_action(self):
        pass

    def up_action(self):
        pass

    def down_action(self):
        pass

if __name__ == "__main__":
    control = Control()
    year = int(input("year: "))
    month = int(input("month: "))
    day = int(input("day: "))
    hours = int(input("hours: "))
    minutes = int(input("minutes: "))
    seconds = int(input("seconds: "))
    control.rtc.set_time(year, month, day, hours, minutes, seconds)
    try:
        while True:
            sleep(1)  # Add a small delay to prevent CPU overuse
            print(control.rtc.read_time())
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        control.buttons.cleanup()