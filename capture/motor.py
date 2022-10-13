import RPi.GPIO as GPIO

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

while True:
    PWM1.ChangeDutyCycle(25)
    PWM2.ChangeDutyCycle(25)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
