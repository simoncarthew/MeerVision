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
    def __init__(self, fps = 10,debug = True):
        # initialise peripherals
        self.camera = Camera()
        self.lcd = LCD()
        self.rtc = RTC()
        self.buttons = Buttons(pin_ok=23, pin_up=27, pin_down=22, pin_back=24)

        # set fps
        self.fps = fps
        
        # over ride the burron actions
        self.buttons.ok_action = self.ok_action
        self.buttons.back_action = self.back_action
        self.buttons.up_action = self.up_action
        self.buttons.down_action = self.down_action

        # global button pressed states
        self.pressed = {"ok":False,"up":False,"down":False,"back":False}

        # set menu items
        self.menu_items = ["HOME", "SINGLE CAPTURE", "DEPLOY", "PROCESS", "TESTING", "SETTINGS"]

        # set current menu_item index
        self.menu_index = 0

    def wait_button(self):
        while True:
            if self.pressed != {"ok":False,"up":False,"down":False,"back":False}:
                return
            else:
                sleep(1/self.fps)

    def ok_action(self):
        self.pressed["ok"] = True

    def back_action(self):
        self.pressed["back"] = True

    def up_action(self):
        self.pressed["up"] = True

    def down_action(self):
        self.pressed["down"] = True

if __name__ == "__main__":
    # initialize main control
    control = Control()

    try:
        while True:
            # display current menu item
            blue_text = ""
            if control.menu_index == 0:
                current_time = control.rtc.read_time()
                blue_text = f"{current_time['hours']}:{current_time['minutes']}:{current_time['seconds']}"
            control.lcd.centered_text(control.menu_items[control.menu_index],blue_text)
            
            # check if a button has been pressed
            control.wait_button()

            if control.pressed["down"]:
                control.menu_index = (control.menu_index + 1) % len(control.menu_items)
            elif control.pressed["up"]:
                if control.menu_index == 0:
                    control.menu_index = len(control.menu_items) - 1
                else:
                    control.menu_index = (control.menu_index - 1) % len(control.menu_items)

            # set buttons states back
            control.pressed = {"ok":False,"up":False,"down":False,"back":False}
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        control.buttons.cleanup()