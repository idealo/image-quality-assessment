try:
    import RPi.GPIO as GPIO
except RuntimeError:
    print("Error importing RPi.GPIO!  This is probably because you need superuser privileges.  You can achieve this by using 'sudo' to run your script")

print("Initial mode: {}".format(GPIO.getmode()))
GPIO.setmode(GPIO.BCM)
print("Final mode: {}".format(GPIO.getmode()))

channel = 18
GPIO.setup(channel, GPIO.OUT, initial=GPIO.HIGH)
input("Take reading, then press enter")

import code
code.interact(local=locals())
