import RPi.GPIO as GPIO
import time

class Buttons:
    def __init__(self, pin_ok, pin_back, pin_up, pin_down, bounce_time=300):
        # Set the GPIO mode
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)  # Disable warnings

        # Store pin numbers
        self.pin_ok = pin_ok
        self.pin_back = pin_back
        self.pin_up = pin_up
        self.pin_down = pin_down

        # Set up pins for input reading
        GPIO.setup(self.pin_ok, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.pin_back, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.pin_up, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(self.pin_down, GPIO.IN, pull_up_down=GPIO.PUD_UP)

        # Set debounce time
        self.bounce_time = bounce_time

        # Small delay to allow GPIO to settle
        time.sleep(0.1)

        # Set up event detection
        try:
            GPIO.add_event_detect(self.pin_ok, GPIO.FALLING, callback=self.on_ok_pressed, bouncetime=self.bounce_time)
            GPIO.add_event_detect(self.pin_back, GPIO.FALLING, callback=self.on_back_pressed, bouncetime=self.bounce_time)
            GPIO.add_event_detect(self.pin_up, GPIO.FALLING, callback=self.on_up_pressed, bouncetime=self.bounce_time)
            GPIO.add_event_detect(self.pin_down, GPIO.FALLING, callback=self.on_down_pressed, bouncetime=self.bounce_time)
        except RuntimeError as e:
            print(f"Error setting up event detection: {e}")

    def get_button_states(self):
        return {
            "ok": GPIO.input(self.pin_ok),
            "up": GPIO.input(self.pin_up),
            "down": GPIO.input(self.pin_down),
            "back": GPIO.input(self.pin_back)
        }

    def on_ok_pressed(self, channel):
        print("OK button pressed")
        self.ok_action()

    def on_back_pressed(self, channel):
        print("Back button pressed")
        self.back_action()

    def on_up_pressed(self, channel):
        print("Up button pressed")
        self.up_action()

    def on_down_pressed(self, channel):
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
        GPIO.cleanup()