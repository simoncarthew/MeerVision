from luma.oled.device import ssd1306
from luma.core.interface.serial import i2c
from PIL import Image, ImageDraw, ImageFont

class LCD:
    def __init__(self):
        self.serial = i2c(port=1, address=0x3C)
        self.device = ssd1306(self.serial)
        self.font = ImageFont.load_default()

    def display_text(self, text):
        # create the image
        image = Image.new('1', (self.device.width, self.device.height), 1)
        draw = ImageDraw.Draw(image)
        
        # add text
        draw.text((0, 0), text, font=self.font, fill=0)
        
        # display text
        self.device.display(image)