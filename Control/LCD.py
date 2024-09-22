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
    
    def display_centered_text(self, text):
        # create blank background
        image = Image.new('1', (self.device.width, self.device.height), "blue")
        draw = ImageDraw.Draw(image)

        # get text size to center
        text_width, text_height = draw.textsize(text, font=self.font)
        x = (self.device.width - text_width) // 2
        y = (self.device.height - text_height) // 2

        # draw the text
        draw.text((x, y), text, font=self.font, fill="yellow")

        # display final image
        self.device.display(image) 

    def display_scroll_wheel_number(self,number):
        # create blank background
        image = Image.new('1', (self.device.width, self.device.height), "blue")
        draw = ImageDraw.Draw(image)

        # get arrow info
        arrow_up = "^"
        arrow_down = "v"
        arrow_size = draw.textsize(arrow_up, font=self.font)

        # calculaate the positions on th eback
        arrow_x = (self.device.width - arrow_size[0]) // 2
        number_width, number_height = draw.textsize(str(number), font=self.font)
        number_x = (self.device.width - number_width) // 2
        number_y = (self.device.height - number_height) // 2

        # Ddraw the arroes and number
        draw.text((arrow_x, 0), arrow_up, font=self.font, fill="yellow")  # top arrow
        draw.text((arrow_x, self.device.height - arrow_size[1]), arrow_down, font=self.font, fill="yellow")  # bottom arrow
        draw.text((number_x, number_y), str(number), font=self.font, fill="yellow")  # center number

        # display the image
        self.device.display(image)  # Convert to monochrome for display

    def display_scroll_wheel_text(self,text):
        # create blank background
        image = Image.new('1', (self.device.width, self.device.height), "blue")
        draw = ImageDraw.Draw(image)

        # get arrow info
        arrow_up = "^"
        arrow_down = "v"
        arrow_size = draw.textsize(arrow_up, font=self.font)

        # calculate the positions of the arrows and text
        arrow_x = (self.device.width - arrow_size[0]) // 2
        text_width, text_height = draw.textsize(text, font=self.font)
        text_x = (self.device.width - text_width) // 2
        text_y = (self.device.height - text_height) // 2

        # draw the arrows and text
        draw.text((arrow_x, 0), arrow_up, font=self.font, fill="yellow")  # top arrow
        draw.text((arrow_x, self.device.height - arrow_size[1]), arrow_down, font=self.font, fill="yellow")  # bottom arrow
        draw.text((text_x, text_y), text, font=self.font, fill="yellow")  # center text

        # display the image
        self.device.display(image)  # Convert to monochrome for display
