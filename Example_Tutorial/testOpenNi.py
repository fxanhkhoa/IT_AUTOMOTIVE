from openni import openni2
from openni import _openni2 as c_api
import numpy as np
import cv2

openni2.initialize()     # can also accept the path of the OpenNI redistribution


dev = openni2.Device.open_any()
print(dev.get_device_info())

rgb_stream = dev.create_color_stream()

#depth_stream = dev.create_depth_stream()
#depth_stream.start()
print('The rgb video mode is', rgb_stream.get_video_mode()) # Checks rgb video configuration
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240, fps=30))

## Start the streams
rgb_stream.start()
while True:
	frame = rgb_stream.read_frame()
	frame_data = frame.get_buffer_as_uint16()
	bgr   = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(240,320,3)
	rgb   = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
	cv2.imshow('frame', rgb)
	#depth_stream.stop()
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

openni2.unload()
cv2.destroyAllWindows()
