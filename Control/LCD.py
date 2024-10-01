from luma.oled.device import ssd1306
from luma.core.interface.serial import i2c
from PIL import Image, ImageDraw, ImageFont
import time

class LCD:
    def __init__(self):
        self.serial = i2c(port=1, address=0x3C)
        self.device = ssd1306(self.serial)
        self.font = ImageFont.load_default()

    def centered_text(self, yellow_text, blue_text):
        # Create blank image with black background
        image = Image.new('1', (self.device.width, self.device.height), 0)
        draw = ImageDraw.Draw(image)

        # Yellow section at the top
        yellow_height = 16
        
        # Use textbbox to calculate the bounding box of the yellow text
        yellow_bbox = draw.textbbox((0, 0), yellow_text, font=self.font)
        yellow_text_width = yellow_bbox[2] - yellow_bbox[0]
        yellow_text_height = yellow_bbox[3] - yellow_bbox[1]

        yellow_x = (self.device.width - yellow_text_width) // 2
        yellow_y = (yellow_height - yellow_text_height) // 2

        # Blue section at the bottom
        blue_bbox = draw.textbbox((0, 0), blue_text, font=self.font)
        blue_text_width = blue_bbox[2] - blue_bbox[0]
        blue_text_height = blue_bbox[3] - blue_bbox[1]

        blue_x = (self.device.width - blue_text_width) // 2
        blue_y = yellow_height + (self.device.height - yellow_height - blue_text_height) // 2

        # Draw the yellow and blue texts
        draw.text((yellow_x, yellow_y), yellow_text, font=self.font, fill=1)  # Yellow part
        draw.text((blue_x, blue_y), blue_text, font=self.font, fill=1)  # Blue part

        # Display the image
        self.device.display(image)

    def scroll_wheel(self, yellow_text, content):
        # Create blank image with black background
        image = Image.new('1', (self.device.width, self.device.height), 0)
        draw = ImageDraw.Draw(image)

        # Yellow section for additional text
        yellow_height = 16
        
        # Use textbbox to calculate the bounding box of the yellow text
        yellow_bbox = draw.textbbox((0, 0), yellow_text, font=self.font)
        yellow_text_width = yellow_bbox[2] - yellow_bbox[0]
        yellow_text_height = yellow_bbox[3] - yellow_bbox[1]

        yellow_x = (self.device.width - yellow_text_width) // 2
        yellow_y = (yellow_height - yellow_text_height) // 2

        # Blue section for scroll wheel (arrows and content)
        arrow_up = "^"
        arrow_down = "v"
        arrow_size = draw.textbbox((0, 0), arrow_up, font=self.font)

        # Positions for arrows and content in the blue section
        blue_start_y = yellow_height  # Blue section starts right after yellow section
        arrow_x = (self.device.width - arrow_size[2]) // 2
        content_bbox = draw.textbbox((0, 0), str(content), font=self.font)
        content_width = content_bbox[2] - content_bbox[0]
        content_height = content_bbox[3] - content_bbox[1]
        content_x = (self.device.width - content_width) // 2
        content_y = blue_start_y + (self.device.height - blue_start_y - content_height) // 2

        # Draw yellow text
        draw.text((yellow_x, yellow_y), yellow_text, font=self.font, fill=1)  # Yellow part

        # Draw arrows and content in the blue section
        draw.text((arrow_x, blue_start_y), arrow_up, font=self.font, fill=1)  # Top arrow
        draw.text((arrow_x, self.device.height - arrow_size[1]), arrow_down, font=self.font, fill=1)  # Bottom arrow
        draw.text((content_x, content_y), str(content), font=self.font, fill=1)  # Center content (number or text)

        # Display the image
        self.device.display(image)

if __name__ == "__main__":
    lcd = LCD()
    lcd.scroll_wheel("hi", 10)
    time.sleep(10)
    lcd.scroll_wheel("hi", "hello")
    time.sleep(10)