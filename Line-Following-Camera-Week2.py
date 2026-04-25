from picamera2 import Picamera2
import RPi.GPIO as GPIO
import cv2
import numpy as np
import time

#Project Week 2 Code
#Line Following via Pi Camera

# Camera Setup
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (320,240), "format":"RGB888"}
    )
)
picam2.start()

WIDTH = 320
CENTER = WIDTH // 2

# PID Parameters
KP = 0.45
KI = 0.0004
KD = 0.14

prev_error = 0
integral = 0
prev_time = time.time()

last_error = 0

# MOTOR SETUP
GPIO.setmode(GPIO.BCM)

ENA = 17
IN1 = 27
IN2 = 22
ENB = 5
IN3 = 6
IN4 = 13

GPIO.setup([ENA,IN1,IN2,ENB,IN3,IN4],GPIO.OUT)

pwmA = GPIO.PWM(ENA,1000)
pwmB = GPIO.PWM(ENB,1000)

pwmA.start(0)
pwmB.start(0)

MAX_SPEED = 60

# MOTOR FUNCTION
def set_motor(left,right):

    if left >= 0:
        GPIO.output(IN1,GPIO.LOW)
        GPIO.output(IN2,GPIO.HIGH)
    else:
        GPIO.output(IN1,GPIO.HIGH)
        GPIO.output(IN2,GPIO.LOW)

    if right >= 0:
        GPIO.output(IN3,GPIO.LOW)
        GPIO.output(IN4,GPIO.HIGH)
    else:
        GPIO.output(IN3,GPIO.HIGH)
        GPIO.output(IN4,GPIO.LOW)

    pwmA.ChangeDutyCycle(min(abs(left),100))
    pwmB.ChangeDutyCycle(min(abs(right),100))

# MAIN LOOP
try:

    while True:

        frame = picam2.capture_array()

        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)

        _,thresh = cv2.threshold(
            blur,
            80,
            255,
            cv2.THRESH_BINARY_INV
        )

        roi = thresh[130:200,:]

        contours,_ = cv2.findContours(
            roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:

            c = max(contours,key=cv2.contourArea)

            if cv2.contourArea(c) > 200:

                M = cv2.moments(c)
                cx = int(M["m10"]/M["m00"])

                error = cx - CENTER
                abs_error = abs(error)

                last_error = error

                # MAXIMUM SHARP CORNER
                if abs_error > 120:

                    if error > 0:
                        set_motor(-100,100)
                    else:
                        set_motor(100,-100)

                    continue
                
                # STRONG CORNER
                if abs_error > 70:

                    if error > 0:
                        set_motor(-70,90)
                    else:
                        set_motor(90,-70)

                    continue

                # NORMAL PID CONTROL
                current_time = time.time()
                dt = current_time - prev_time

                integral += error * dt
                derivative = (error - prev_error) / dt

                correction = (
                    KP * error +
                    KI * integral +
                    KD * derivative
                )

                prev_error = error
                prev_time = current_time

                correction = max(-120,min(120,correction))

                left = MAX_SPEED - correction
                right = MAX_SPEED + correction

                set_motor(left,right)

        else:

            # LOST LINE RECOVERY
            if last_error > 0:
                set_motor(-80,80)
            else:
                set_motor(80,-80)

        cv2.imshow("Camera",frame)

        
        cv2.imshow("Threshold",thresh)

        if cv2.waitKey(1) == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:

    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()
    picam2.stop()
    cv2.destroyAllWindows()
    print("Program Successfully Terminated")
