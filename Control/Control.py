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
        
        # Override button actions
        self.buttons.ok_action = self.ok_action
        self.buttons.back_action = self.back_action
        self.buttons.up_action = self.up_action
        self.buttons.down_action = self.down_action

    def ok_action(self):
        print("Control: OK action")
        # Add your custom OK action here

    def back_action(self):
        print("Control: Back action")
        # Add your custom Back action here

    def up_action(self):
        print("Control: Up action")
        # Add your custom Up action here

    def down_action(self):
        print("Control: Down action")
if __name__ == "__main__":
    control = Control()
    try:
        while True:
            # print(control.buttons.get_button_states())
            sleep(0.1)  # Add a small delay to prevent CPU overuse
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        control.buttons.cleanup()

    # while True:
    #     sleep(0.5)
    #     control.camera.capture("Control/test.jpg")