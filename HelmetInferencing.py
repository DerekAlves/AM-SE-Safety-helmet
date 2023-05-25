# Run the person detection model
# This version reads the images from the ov2640 camera on the esp32-cam board
# with minor changes this also works for the m5 timer camera
 
import sys
import microlite
from machine import Pin,PWM
import image_arr


# initialize the flash-light LED, it is connected to GPIO 4
flash_light = PWM(Pin(4))
# switch it off
flash_light.duty(0)

# change for m5 timer camera
# # initialize the flash-light LED, it is connected to GPIO 4
#  flash_light = Pin(2,Pin.OUT)
# switch it off
# flash_light.off()

mode = 1
test_image = bytearray (9600)
test_image = image_arr.h1
#print(test_image)

def handle_output(helmet):
	if helmet > 5:
		flash_light.duty(5)
	else:
		flash_light.duty(0)
        
def input_callback (microlite_interpreter):    
	inputTensor = microlite_interpreter.getInputTensor(0)
	for i in range (0, len(test_image)):
		inputTensor.setValue(i, test_image[i]-127)
	print ("setup %d bytes on the inputTensor." % (len(test_image)))

def output_callback (microlite_interpreter):
	outputTensor = microlite_interpreter.getOutputTensor(0)
	print(outputTensor)
	helmet = outputTensor.getValue(0)
	noHelmet = outputTensor.getValue(1)
	print ("'Helmet' = %d, 'No Helmet' = %d" % (helmet, noHelmet))
	handle_output(helmet)

# read the model
helmet_model_file = open ('safe.tflite', 'rb')
helmet_model = bytearray (113016)
helmet_model_file.readinto(helmet_model)
helmet_model_file.close()

# create the interpreter
interp = microlite.interpreter(helmet_model,800000, input_callback, output_callback)

# Permanently read images from the camera and pass them into the model for
# inference

while True:
	print('h1')
	interp.invoke()
	print('n1')
	test_image = image_arr.n1
	interp.invoke()
	print('h2')
	test_image = image_arr.h2
	interp.invoke()
	print('n2')
	test_image = image_arr.n2
	interp.invoke()
	test_image = image_arr.h1


