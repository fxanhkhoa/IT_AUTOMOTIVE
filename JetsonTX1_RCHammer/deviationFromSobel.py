import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

global used_warped
global used_ret
global left_curve
global right_curve


class deviation:

	y = 150
	factor = 2.5
	premiddle = 0

	def __init__(self):
		self.left_fit_his = []
		self.right_fit_his = []
		return
	
	######## Camera callibration
	def cal_undistort(self, img, objpoints, imgpoints):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,None)
		undist = cv2.undistort(img, mtx, dist, None, mtx)
		return undist, mtx, dist

	def collect_callibration_points(self):
		objpoints = []
		imgpoints = []
   
		images = glob.glob('./camera_cal/calibration*.jpg')
		objp = np.zeros((6*9,3), np.float32)
		objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1, 2)
		
		for fname in images:
			img = mpimg.imread(fname)

			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

			if ret == True:
				imgpoints.append(corners)
				objpoints.append(objp)
			
		return imgpoints, objpoints

	def compare_images(self, image1, image2, image1_exp="Image 1", image2_exp="Image 2"):
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
		f.tight_layout()
		ax1.imshow(image1)
		ax1.set_title(image1_exp, fontsize=50)
		ax2.imshow(image2)
		ax2.set_title(image2_exp, fontsize=50)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		#plt.show()

	######## Gradient Thresholds

	def abs_sobel_thresh(self, image, orient='x', sobel_kernel=3, thresh=(0, 255)):
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		isX = True if orient == 'x' else False
		sobel = cv2.Sobel(gray, cv2.CV_64F, isX, not isX)
		abs_sobel = np.absolute(sobel)
		scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 
		grad_binary = np.zeros_like(scaled_sobel)
		grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
	   
		return grad_binary
		
	def mag_thresh(self, image, sobel_kernel=3, mag_thresh=(0, 255)):
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		abs_sobel = np.sqrt(sobelx**2 + sobely**2)
		scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel)) 
		mag_binary = np.zeros_like(scaled_sobel)
		mag_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

		return mag_binary
		
	def dir_threshold(self, image, sobel_kernel=3, thresh=(0, np.pi/2)):
		gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
		sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
		abs_sobelx = np.absolute(sobelx)
		abs_sobely = np.absolute(sobely)
		grad_dir = np.arctan2(abs_sobely, abs_sobelx)
		dir_binary = np.zeros_like(grad_dir)
		dir_binary[(grad_dir >= thresh[0]) & (grad_dir <= thresh[1])] = 1

		return dir_binary
		
	def apply_thresholds(self, image, ksize=3):
		gradx = self.abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=(20, 100))
		grady = self.abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=(20, 100))
		mag_binary = self.mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30, 100))
		dir_binary = self.dir_threshold(image, sobel_kernel=ksize, thresh=(0.7, 1.3))

		combined = np.zeros_like(dir_binary)
		combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
		
		return combined
		
	##### Color Threshold #######


	def apply_color_threshold(self, image):
		hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
		s_channel = hls[:,:,2]
		s_thresh_min = 170
		s_thresh_max = 255
		s_binary = np.zeros_like(s_channel)
		s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

		return s_binary

	### Combine Color and Gradient ###

	def combine_threshold(self, s_binary, combined):
		combined_binary = np.zeros_like(combined)
		combined_binary[(s_binary == 1) | (combined == 1)] = 1

		return combined_binary

	### Perspective Transform ###

	def warp(self, img):

		imshape = img.shape
		img_size = (img.shape[1], img.shape[0])
		
		src = np.float32(
			[[0.7*imshape[1], 0.45*imshape[0]], 
			  [0.8 * imshape[1],1*imshape[0]], 
			  [0.2 * imshape[1],1*imshape[0]], 
			  [0.3*imshape[1], 0.45*imshape[0]]])
		
		dst = np.float32(
			[[0.6*img.shape[1],0*imshape[0]], 
			  [0.6*img.shape[1],1*img.shape[0]], 
			  [0.4*img.shape[1],1*img.shape[0]], 
			  [0.4*img.shape[1],0*imshape[0]]])
		
		M = cv2.getPerspectiveTransform(src, dst)
		Minv = cv2.getPerspectiveTransform(dst, src)
		
		binary_warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
		#cv2.circle(img,(int(0.7*imshape[1]), int(0.7*imshape[0])), 10, (255,255,255), -1)
		#cv2.circle(img,(int(imshape[1]),int(1*imshape[0])), 10, (255,255,255), -1)
		#cv2.circle(img,(0,imshape[0]), 10, (255,255,255), -1)
		#cv2.circle(img,(int(0.3*imshape[1]),int(0.7*imshape[0])), 10, (255,255,255), -1)

		#cv2.circle(img,(685,450), 10, (255,255,255), -1)
		#cv2.circle(img,(1090, 710), 10, (255,255,255), -1)
		#cv2.circle(img,(220, 710), 10, (255,255,255), -1)
		#cv2.circle(img,(595, 450), 10, (255,255,255), -1)

		#cv2.circle(img,(900, 0), 10, (255,255,255), -1)
		#cv2.circle(img,(900, 710), 10, (255,255,255), -1)
		#cv2.circle(img,(250, 710), 10, (255,255,255), -1)
		#cv2.circle(img,(250, 0), 10, (255,255,255), -1)

		#cv2.imshow('draw', img)
		return binary_warped, Minv

	### Histogram ###

	def get_histogram(self, binary_warped):
		histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
		
		return histogram

	def compare_plotted_images(self, image1, image2, image1_exp="Image 1", image2_exp="Image 2"):
		f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
		f.tight_layout()
		ax1.imshow(image1)
		ax1.plot([214, 340], [140, 222], color='r', linewidth="5")
		ax1.plot([340, 68], [222, 222], color='r', linewidth="5")
		ax1.plot([220, 185], [222, 140], color='r', linewidth="5")
		ax1.plot([185, 214], [140, 140], color='r', linewidth="5")
		ax1.set_title(image1_exp, fontsize=50)
		ax2.imshow(image2)
		ax2.plot([282, 282], [0, 222], color='r', linewidth="5")
		ax2.plot([282, 78], [222, 222], color='r', linewidth="5")
		ax2.plot([78, 78], [222, 0], color='r', linewidth="5")
		ax2.plot([78, 282], [0, 0], color='r', linewidth="5")
		ax2.set_title(image2_exp, fontsize=50)
		plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
		plt.show()


	### Sliding Window ###

	def slide_window(self, binary_warped, histogram):
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		midpoint = np.int(histogram.shape[0]/2)
		leftx_base = np.argmax(histogram[:midpoint])
		rightx_base = np.argmax(histogram[midpoint:]) + midpoint

		nwindows = 9
		window_height = np.int(binary_warped.shape[0]/nwindows)
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		leftx_current = leftx_base
		rightx_current = rightx_base
		margin = 50
		minpix = 20
		left_lane_inds = []
		right_lane_inds = []

		for window in range(nwindows):
			win_y_low = binary_warped.shape[0] - (window+1)*window_height
			win_y_high = binary_warped.shape[0] - window*window_height
			win_xleft_low = leftx_current - margin
			win_xleft_high = leftx_current + margin
			win_xright_low = rightx_current - margin
			win_xright_high = rightx_current + margin
			cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
			(0,255,0), 2) 
			cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
			(0,255,0), 2) 
			good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
			good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
			(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
			left_lane_inds.append(good_left_inds)
			right_lane_inds.append(good_right_inds)
			if len(good_left_inds) > minpix:
				leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
			if len(good_right_inds) > minpix:        
				rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

		left_lane_inds = np.concatenate(left_lane_inds)
		right_lane_inds = np.concatenate(right_lane_inds)

		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds] 
		
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		if (leftx.size > 0):
		#if (1):
			left_fit = np.polyfit(lefty, leftx, 2)
			left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
			self.left_fit_his = left_fit
		else:
			print(self.left_fit_his)
			left_fit = self.left_fit_his
		if (rightx.size > 0):
		#if (1):
			right_fit = np.polyfit(righty, rightx, 2)
			right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
			self.right_fit_his = right_fit
		else:
			right_fit = self.right_fit_his
		

		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		#cv2.imshow('sliding window',out_img)
		#plt.imshow(out_img)
		#plt.plot(left_fitx, ploty, color='yellow')
		#plt.plot(right_fitx, ploty, color='yellow')
		#plt.xlim(0, 1280)
		#plt.ylim(720, 0)
		
		return ploty, left_fit, right_fit

	### Skipping Slinding Window ###

	def skip_sliding_window(self, binary_warped, left_fit, right_fit):
		nonzero = binary_warped.nonzero()
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		margin = 100
		left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
		left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
		left_fit[1]*nonzeroy + left_fit[2] + margin))) 

		right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
		right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
		right_fit[1]*nonzeroy + right_fit[2] + margin)))  

		leftx = nonzerox[left_lane_inds]
		lefty = nonzeroy[left_lane_inds] 
		rightx = nonzerox[right_lane_inds]
		righty = nonzeroy[right_lane_inds]
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)
		ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		mid_fitx = (left_fitx + right_fitx) / 2
		
		################################ 
		## Visualization
		################################ 
		
		out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
		window_img = np.zeros_like(out_img)
		out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
		out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

		left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
		left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
									  ploty])))])
		left_line_pts = np.hstack((left_line_window1, left_line_window2))
		right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
		right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
									  ploty])))])
		right_line_pts = np.hstack((right_line_window1, right_line_window2))

		cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
		cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
		result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

		#cv2.imshow('skipped, sliding', result)
		#plt.imshow(result)
		#plt.plot(left_fitx, ploty, color='yellow')
		#plt.plot(right_fitx, ploty, color='yellow')
		#plt.xlim(0, 1280)
		#plt.ylim(720, 0)
		
		ret = {}
		ret['leftx'] = leftx
		ret['rightx'] = rightx
		ret['left_fitx'] = left_fitx
		ret['right_fitx'] = right_fitx
		ret['ploty'] = ploty
		
		return ret

	### Measuring Curvature ###

	def measure_curvature(self, ploty, lines_info):
		ym_per_pix = 30/720 
		xm_per_pix = 3.7/700 

		leftx = lines_info['left_fitx']
		rightx = lines_info['right_fitx']

		leftx = leftx[::-1]  
		rightx = rightx[::-1]  

		y_eval = np.max(ploty)
		left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
		right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
		left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
		print(left_curverad, 'm', right_curverad, 'm')
		
		return left_curverad, right_curverad

	### Drawing ###
	def draw_lane_lines(self, original_image, warped_image, Minv, draw_info):
		leftx = draw_info['leftx']
		rightx = draw_info['rightx']
		left_fitx = draw_info['left_fitx']
		right_fitx = draw_info['right_fitx']
		ploty = draw_info['ploty']
		
		warp_zero = np.zeros_like(warped_image).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))
		
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

		newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0])) 
		result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)
		
		return result

	# image is expected be in RGB color space
	def select_rgb_white_yellow(self, image): 
		# white color mask
		lower = np.uint8([200, 200, 200])
		upper = np.uint8([255, 255, 255])
		white_mask = cv2.inRange(image, lower, upper)
		# yellow color mask
		lower = np.uint8([0, 190,   190])
		upper = np.uint8([160, 255, 255])
		yellow_mask = cv2.inRange(image, lower, upper)
		# combine the mask
		mask = cv2.bitwise_or(white_mask, yellow_mask)
		masked = cv2.bitwise_and(image, image, mask = mask)

		gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY)
		#cv2.imshow('masked', gray)
		#cv2.imshow('white mask', white_mask)
		#cv2.imshow('yellow mask', yellow_mask)
		
		return gray

	# Calculate Angle
	def GetAngle(self, x, xshape):
		# Calculate Angle
		#x = middle
		#xshape = (image.shape[1] / 2) 
		print('hieu so', x-xshape)
		value = math.atan2((x-xshape), self.y)
		result = value * 180 / math.pi
		result = result * self.factor
		if result < 0:
			result = result + result / 7.5
		print('goc lech = ', result)
		return result

	### Defining image processing method ###

	def process_image(self, image):
		#global used_warped
		#global used_ret

		#imgpoints, objpoints = self.collect_callibration_points()
		#img = mpimg.imread('./camera_cal/calibration3.jpg')
		#undistorted, mtx, dist_coefficients = cal_undistort(img, objpoints, imgpoints)
		#compare_images(img, undistorted, "Original Image", "Undistorted Image")
		
		#Undistort image
		#image, mtx, dist_coefficients = self.cal_undistort(image, objpoints, imgpoints)
		#cv2.imshow('undistort', image)
		
		#################### FOR SOBEL #####################
		# Gradient thresholding
		#gradient_combined = self.apply_thresholds(image)
		#cv2.imshow('Gradient', gradient_combined)
		   
		# Color thresholding
		#s_binary = self.apply_color_threshold(image)
		#compare_images(image, s_binary, "Original Image", "Color Threshold")
		
		# Combine Gradient and Color thresholding
		#combined_binary = self.combine_threshold(s_binary, gradient_combined)
		#compare_images(image, combined_binary, "Original Image", "combined_binary")
		##########################################

		#Binary 
		combined_binary = self.select_rgb_white_yellow(image)
	   
		# Transforming Perspective
		binary_warped, Minv = self.warp(combined_binary)
		#cv2.imshow('binary_warped', binary_warped)
		#compare_plotted_images(image, binary_warped, "Original Image", "Warped Image")
	  
		# Getting Histogram
		histogram = self.get_histogram(binary_warped)
		#plt.plot(histogram)
		#plt.show()
	  
		# Sliding Window to detect lane lines
		ploty, left_fit, right_fit = self.slide_window(binary_warped, histogram)
		
		# Skipping Sliding Window
		ret = self.skip_sliding_window(binary_warped, left_fit, right_fit)

		# Measuring Curvature
		left_curverad, right_curverad = self.measure_curvature(ploty, ret)
		print(left_curverad,' ', right_curverad)
		
		# Get middle
		#middle = max(ret['rightx']) + min(ret['leftx'])
		avr_right = sum(ret['rightx']) / len(ret['rightx'])
		avr_left = sum(ret['leftx']) / len(ret['leftx'])
		middle = (avr_right + avr_left) / 2
		print('mid', middle)
		self.premiddle = middle
		cv2.circle(image,(int(image.shape[1] / 2)  , image.shape[0]), 10, (255,255,255), -1)
		cv2.circle(image, (int(middle), 450), 5, (0,0,255), -1)
		cv2.imshow('output', image)
		
		if (right_curverad == left_curverad):
			if (right_curverad < left_curverad):
				print('lost 1 lane right')
				middle = self.premiddle + (self.premiddle - middle)
			else:
				print('lost 1 lane left')
				middle = self.premiddle - (middle - self.premiddle)
			
		
		if (left_curverad > 0):
			left_curve = left_curverad
		if right_curverad > 0:
			right_curve = right_curverad
		
		# Calculate Angle
		print('premid', self.premiddle)
		angle = self.GetAngle(middle, image.shape[1] / 2)
		#print(right_curve)
		
		# Sanity check: whether the lines are roughly parallel and have similar curvature
		#slope_left = ret['left_fitx'][0] - ret['left_fitx'][-1]
		#slope_right = ret['right_fitx'][0] - ret['right_fitx'][-1]
		#slope_diff = abs(slope_left - slope_right)
		#slope_threshold = 150
		#curve_diff = abs(left_curverad - right_curverad)
		#curve_threshold = 10000
		
		#if (slope_diff > slope_threshold or curve_diff > curve_threshold):
		#	binary_warped = used_warped
		#	ret = used_ret
	   
		# Visualizing Lane Lines Info
		#result = self.draw_lane_lines(image, binary_warped, Minv, ret)
		
		# Annotating curvature 
		#fontType = cv2.FONT_HERSHEY_SIMPLEX
		#curvature_text = 'The radius of curvature = ' + str(round(left_curverad, 3)) + 'm'
		#cv2.putText(result, curvature_text, (30, 60), fontType, 1.5, (255, 255, 255), 3)
	   
		# Annotating deviation
		#deviation_pixels = image.shape[1]/2 - abs(ret['right_fitx'][-1] - ret['left_fitx'][-1])
		#xm_per_pix = 3.7/700 
		#deviation = deviation_pixels * xm_per_pix
		#direction = "left" if deviation < 0 else "right"
		#deviation_text = 'Vehicle is ' + str(round(abs(deviation), 3)) + 'm ' + direction + ' of center'
		#cv2.putText(result, deviation_text, (30, 110), fontType, 1.5, (255, 255, 255), 3)
		
		#used_warped = binary_warped
		#used_ret = ret

		#cv2.imshow('result', result)
		
		return angle
