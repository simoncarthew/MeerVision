from picamera2 import Picamera2
from time import sleep
import os
from luma.oled.device import ssd1306
from luma.core.interface.serial import i2c
from PIL import Image, ImageDraw, ImageFont

class Control:
    def __init__(self, lcd = True):
        # intialise the camera
        self.camera = Picamera2()
        
        # initialise screen
        if lcd:
            self.serial = i2c(port=1, address=0x3C)
            self.device = ssd1306(self.serial)
            self.font = ImageFont.load_default()

    def capture(self, pic_path):
        directory = os.path.dirname(pic_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        self.camera.start()
        sleep(1)
        self.camera.capture_file(pic_path)
        self.camera.stop()

    def display_text(self, text):
        # create the image
        image = Image.new('1', (self.device.width, self.device.height), 1)
        draw = ImageDraw.Draw(image)
        
        # add text
        draw.text((0, 0), text, font=self.font, fill=0)
        
        # display text
        self.device.display(image)

if __name__ == "__main__":
    control = Control(lcd=False)
    # control.display_text("BOOBS")
    # sleep(20)
    while True:
        sleep(0.5)
        control.capture("Control/test.jpg")