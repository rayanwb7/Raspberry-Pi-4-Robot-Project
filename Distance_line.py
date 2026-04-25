import RPi.GPIO as GPIO
import time
#Project Week 1 Code
#Where it can travel at a user inputed distance
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
#Duty Cycle Values
PWM_VALUE=90
speedL=0.9*PWM_VALUE
speedR=PWM_VALUE

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
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)

#specfic distance movement function
def forwardmove_distance(speedL,speedR,distance):
    forwardmove(speedL,speedR)
    time_needed = distance / ((speedL + speedR) / 2 * 0.75)  # Simplified time calculation
    time.sleep(time_needed)
    stopmove()
    total=distance/time_needed
    print("Speed:",total,"cm/s")

try:
    forwardmove(speedL,speedR)
    forwardmove_distance(speedL,speedR,50) #move forward with x distance.

except KeyboardInterrupt:
    pass

finally:
    pwmA.stop()
    pwmB.stop()
    GPIO.cleanup()

    print("Program has ended and GPIO cleared")
