from time import sleep
import  time
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

SGL_CAP_PATH = os.path.join("Control","Images","SingleCapture")
DEP_PATH = os.path.join("Control","Images","Deployments")

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
        self.menu_items = {"home":"HOME", "sgl":"SINGLE CAPTURE", "dep":"DEPLOY", "proc":"PROCESS", "test":"TESTING", "set":"SETTINGS"}

        # set current menu_item index
        self.menu_index = "home"
        self.menu_keys = list(self.menu_items.keys())

    def wait_button(self):
        while True:
            if self.pressed != {"ok":False,"up":False,"down":False,"back":False}:
                return
            else:
                sleep(1/self.fps)
    
    def reset_buttons(self):
        self.pressed = {"ok":False,"up":False,"down":False,"back":False}

    def ok_action(self):
        self.pressed["ok"] = True

    def back_action(self):
        self.pressed["back"] = True

    def up_action(self):
        self.pressed["up"] = True

    def down_action(self):
        self.pressed["down"] = True

    def active_scroll_wheel(self, yellow_text, items):

        # create items list if number range
        if isinstance(items[0], int):
            items = list(range(items[0], items[1] + 1))
        index = 0

        while True:
            # display current item
            self.lcd.scroll_wheel(yellow_text,items[index])

            # wait for the user to make decision
            self.wait_button()

            # action on selected choice
            if self.pressed["ok"]: 
                self.reset_buttons()
                return items[index]
            elif self.pressed["down"]: index -= 1
            elif self.pressed["up"]: index += 1
            elif self.pressed["back"]:
                self.reset_buttons() 
                return None

            # loop round index
            index = index % len(items)

            # reset buttons
            self.reset_buttons()
        pass

    def single_capture(self):
        while True:
            self.lcd.centered_text(self.menu_items["sgl"],"Press OK to capture.")
            
            # wait for button input
            self.wait_button()

            # respond to button input
            if self.pressed["ok"]:
                img_name = self.camera.time_to_path(self.rtc.read_time())
                self.camera.capture(save_path=os.path.join(SGL_CAP_PATH,img_name))
                self.lcd.centered_text(self.menu_items["sgl"],"Captured Successfully")
                sleep(2)
            elif self.pressed["back"]:
                return

            self.reset_buttons()

    def continuous_capture(self):
        print("Entered Continuous")

        if not self.camera.running:
            # get the desired fps
            fps = self.active_scroll_wheel("SELECT FPS",[1,5])

            # get timed or indefinite
            timed = self.active_scroll_wheel("TIMED OR INDEFINITE?",["Timed","Indefinite"])

            # get the actual time if timed
            if timed == "Timed":
                hours = self.active_scroll_wheel("HOURS",[0,1])
                minutes = self.active_scroll_wheel("MINUTES",[1,60])

            # calculate the duration
            duration = (hours * 60 + minutes) * 60
            print(duration)

            # start capturing
            self.camera.start_capture_period(DEP_PATH,self.rtc,fps,duration)

            # reset the buttons
            self.reset_buttons()
        else:
            action = self.active_scroll_wheel("ACTIVE DEPLOYMENT",["End Early","View Progress"])

            if action == "End Early":
                self.camera.stop_capture_period()
                self.lcd.centered_text("","Deployment Ended")
            elif action == "View Progress":
                progress = self.camera.run_time / self.camera.total_duration * 100
                self.lcd.percentage_bar(f"{progress:.2f}%", progress)

if __name__ == "__main__":
    # initialize main control
    control = Control()

    try:
        while True:
            # display current menu item
            blue_text = ""
            if control.menu_index == "home":
                current_time = control.rtc.read_time()
                blue_text = f"{current_time['hours']}:{current_time['minutes']}:{current_time['seconds']}"
            control.lcd.centered_text(control.menu_items[control.menu_index],blue_text)
            
            # wait for button input
            control.wait_button()

            # respond to button input
            if control.pressed["down"]: # move down the menu items
                control.menu_index = control.menu_keys[(control.menu_keys.index(control.menu_index) + 1) % len(control.menu_items)]

            elif control.pressed["up"]: # move up the menu items
                control.menu_index = control.menu_keys[len(control.menu_items) - 1] if control.menu_index == "home" else control.menu_keys[(control.menu_keys.index(control.menu_index) - 1) % len(control.menu_items)]
        
            elif control.pressed["ok"]: # enter desired mode
                control.reset_buttons()
                if control.menu_index == "sgl": control.single_capture()
                if control.menu_index == "dep": control.continuous_capture()
            
            # display current mode
            print(control.menu_items[control.menu_index])

            # set buttons states back
            control.reset_buttons()
    except KeyboardInterrupt:
        print("Program terminated by user")
    finally:
        control.buttons.cleanup()