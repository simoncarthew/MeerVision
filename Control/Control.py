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

        # global button pressed states
        self.ok_pressed = False
        self.up_pressed = False
        self.down_pressed = False
        self.back_pressed = False

        # set menu items
        self.menu_items = ["HOME", "SINGLE CAPTURE", "DEPLOY", "PROCESS", "TESTING", "SETTINGS"]

        # set current menu_item index
        self.menu_index = 0

    def ok_action(self):
        self.ok_pressed = True
        pass

    def back_action(self):
        self.back_pressed = True
        pass

    def up_action(self):
        self.up_pressed = True
        pass

    def down_action(self):
        self.down_pressed = True
        if self.menu_index == 0:
            self.menu_index = (self.menu_index + 1) % len(self.menu_items)
        pass

if __name__ == "__main__":
    # initialize main control
    control = Control()

    try:
        while True:
            # display current menu item
            blue_text = ""
            if control.menu_index == 0:
                time = control.rtc.read_time()
                blue_text = f"{time.hours}:{time.minutes}:{time.seconds}"
            control.lcd.centered_text(control.menu_items[control.menu_index],blue_text)
            sleep(0.5)
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        control.buttons.cleanup()