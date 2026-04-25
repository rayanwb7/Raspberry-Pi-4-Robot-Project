import RPi.GPIO as GPIO
import time

#Project Week 1 Code
#Where the robot can turn at a certain user inputed angle

GPIO.setmode(GPIO.BCM)
#BCM Pins on Raspberry Pi
ENA = 17
IN1 = 27
IN2 = 22
ENB = 5
IN3 = 6
IN4 = 13

GPIO.setup([ENA, IN1, IN2, ENB, IN3, IN4], GPIO.OUT)

#PWM Setup with 1kHz Duty Cycle Frequency
pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
#initialize both PWM with 0
pwmA.start(0)
pwmB.start(0)
#Right Turn
#90 degrees: 0.93 and 0.8 with 2.75s
#180 degrees: 0.87 and 0.8 with 2.5s
#360 degrees: same speed with 2.5s

#Left Turn
#90 degrees: base and 0.8 with 2.9s
#180 degrees: base and 0.8 with 2.75s
#360 degrees: same speed with 2.75s

#calibration constants
time_turn_360deg=2.9 #time taken to turn 360 degrees (can be changed according to test results)
PWM_VALUE=90 #duty cycle value in between 0-100
speedL=0.93*PWM_VALUE #left side motor speed
speedR=0.8*PWM_VALUE #right side motor speed to adjust for uneven motor power

# Movement functions
def forwardmove(speedL,speedR):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speedL)
    pwmB.ChangeDutyCycle(speedR)

def stopmove():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)

def turnleft(speedL,speedR):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(speedL)
    pwmB.ChangeDutyCycle(speedR)

def backwardmove(speedL,speedR):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmA.ChangeDutyCycle(speedL)
    pwmB.ChangeDutyCycle(speedR)

def turnright(speedL,speedR):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwmA.ChangeDutyCycle(speedL)
    pwmB.ChangeDutyCycle(speedR)

def angleRight(degrees):
    turn_time=(degrees/360)*time_turn_360deg
    turnright(speedL,speedR)
    time.sleep(turn_time)
    stopmove()

def angleLeft(degrees):
    turn_time=(degrees/360)*time_turn_360deg
    turnleft(speedL,speedR)
    time.sleep(turn_time)
    stopmove()

try:
    #Placeholder values for testing
    angleLeft(90)
    time.sleep(1)
    #angleRight(90)
    time.sleep(1)

except KeyboardInterrupt:
    pass

finally:
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()

    print("Program has ended and GPIO cleaned")
