import RPi.GPIO as GPIO
import time
from picamera2 import Picamera2, Preview

#Purpose of this code it is to test out the functionality of Pi Camera

picam2=Picamera2()
picam2.start_preview(Preview.QTGL)
preview_config=picam2.create_preview_configuration()
picam2.configure(preview_config)

picam2.start()
time.sleep(2)
