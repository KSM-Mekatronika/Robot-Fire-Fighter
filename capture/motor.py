import RPi.GPIO as GPIO
import time
IN1 = 14
IN2 = 15
IN3 = 20
IN4 = 21
ENA = 18
ENB = 12

GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(ENB, GPIO.OUT)

GPIO.output(IN1, GPIO.LOW)
GPIO.output(IN2, GPIO.LOW)
GPIO.output(IN3, GPIO.LOW)
GPIO.output(IN4, GPIO.LOW)

PWM1 = GPIO.PWM(ENA, 1000)
PWM2 = GPIO.PWM(ENB, 1000)
PWM1.start(0)
PWM2.start(0)


def move_forward():
    PWM1.ChangeDutyCycle(40)
    PWM2.ChangeDutyCycle(40)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)



def move_left():
    PWM1.ChangeDutyCycle(30)
    PWM2.ChangeDutyCycle(60)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)



def move_right():
    PWM1.ChangeDutyCycle(60)
    PWM2.ChangeDutyCycle(30)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)



while True:
    move_forward()
    time.sleep(5)
    move_left()
    time.sleep(5)
    move_right()
    time.sleep(5)
