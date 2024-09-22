from gpiozero import Button
from time import sleep

class Buttons:
    def __init__(self, pin_ok, pin_back, pin_up, pin_down, bounce_time=0.3):
        # Create Button objects
        self.btn_ok = Button(pin_ok, bounce_time=bounce_time)
        self.btn_back = Button(pin_back, bounce_time=bounce_time)
        self.btn_up = Button(pin_up, bounce_time=bounce_time)
        self.btn_down = Button(pin_down, bounce_time=bounce_time)

        # Set up event handlers
        self.btn_ok.when_pressed = self.on_ok_pressed
        self.btn_back.when_pressed = self.on_back_pressed
        self.btn_up.when_pressed = self.on_up_pressed
        self.btn_down.when_pressed = self.on_down_pressed

    def get_button_states(self):
        return {
            "ok": not self.btn_ok.is_pressed,
            "up": not self.btn_up.is_pressed,
            "down": not self.btn_down.is_pressed,
            "back": not self.btn_back.is_pressed
        }

    def on_ok_pressed(self):
        print("OK button pressed")
        self.ok_action()

    def on_back_pressed(self):
        print("Back button pressed")
        self.back_action()

    def on_up_pressed(self):
        print("Up button pressed")
        self.up_action()

    def on_down_pressed(self):
        print("Down button pressed")
        self.down_action()

    # Override these methods to define custom actions for each button
    def ok_action(self):
        pass

    def back_action(self):
        pass

    def up_action(self):
        pass

    def down_action(self):
        pass

    def cleanup(self):
        """Clean up the GPIO pins when done."""
        # gpiozero handles cleanup automatically, but we'll keep this method for consistency
        pass