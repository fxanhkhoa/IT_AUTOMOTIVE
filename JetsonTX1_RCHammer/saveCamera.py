import cv2
import numpy as np
from openni import openni2
from openni import _openni2 as c_api

out = cv2.VideoWriter('00202_out.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (640,480))
openni2.initialize()

dev = openni2.Device.open_any()
print(dev.get_device_info())

rgb_stream = dev.create_color_stream()

print('The rgb video mode is', rgb_stream.get_video_mode()) # Checks rgb video configuration
rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))

## Start the streams
rgb_stream.start()
	
	

while True:
	bgr = np.fromstring(rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(480,640,3)
	img = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
	img = cv2.flip( img, 1 )
	out.write(img)
	cv2.imshow('img',img)
	
	if cv2.waitKey(33)& 0xFF == ord('q'):
		break
        
cap.release()
cv2.destroyAllWindows()
