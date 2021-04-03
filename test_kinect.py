from __future__ import print_function

import os, glob, time, argparse, pdb, cv2
#import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from functions import *
from networks import ResnetConditionHR

import sys
sys.path.insert(1, '../pyKinectAzure/')
from pyKinectAzure import pyKinectAzure, _k4a

# Kinect Driver Path
modulePath = 'C:\\Program Files\\Azure Kinect SDK v1.4.1\\sdk\\windows-desktop\\amd64\\release\\bin\\k4a.dll' 
bodyTrackingModulePath = 'C:\\Program Files\\Azure Kinect Body Tracking SDK\\sdk\\windows-desktop\\amd64\\release\\bin\\k4abt.dll'


def depth_to_mask(scr_Image):
	height = scr_Image.shape[0]
	width = scr_Image.shape[1]
	# 生成和原图一样高度和宽度的矩形（全为0）
	dst_Image = scr_Image

	# 以下是copyTo的算法原理：
	# 先遍历每行每列（如果不是灰度图还需遍历通道，可以事先把mask图转为灰度图）
	for row in range(height):
		for col in range(width):

			# 如果掩图的像素不等于0，则dst(x,y) = scr(x,y)
			if scr_Image[row, col] != 0:
				dst_Image[row, col] = 255
					
			# 如果掩图的像素等于0，则dst(x,y) = 0
			else:
				dst_Image[row, col] = 0
	return dst_Image

torch.set_num_threads(1)
os.environ["CUDA_VISIBLE_DEVICES"]="0"
print('CUDA Device: ' + os.environ["CUDA_VISIBLE_DEVICES"])


"""Parses arguments."""
parser = argparse.ArgumentParser(description='Background Matting.')
parser.add_argument('-m', '--trained_model', type=str, default='real-fixed-cam',choices=['real-fixed-cam', 'real-hand-held', 'syn-comp-adobe'],help='Trained background matting model')
#parser.add_argument('-o', '--output_dir', type=str, required=True,help='Directory to save the output results. (required)')
#parser.add_argument('-i', '--input_dir', type=str, required=True,help='Directory to load input images. (required)')
#parser.add_argument('-tb', '--target_back', type=str,help='Directory to load the target background.')
#parser.add_argument('-b', '--back', type=str,default=None,help='Captured background image. (only use for inference on videos with fixed camera')


args=parser.parse_args()

#input model
model_main_dir='Model/' + args.trained_model + '/';
#input data path
#data_path=args.input_dir

'''
if os.path.isdir(args.target_back):
	args.video=True
	print('Using video mode')
else:
	args.video=False
	print('Using image mode')
	#target background path
	back_img10=cv2.imread(args.target_back); back_img10=cv2.cvtColor(back_img10,cv2.COLOR_BGR2RGB);
	#Green-screen background
	back_img20=np.zeros(back_img10.shape); back_img20[...,0]=120; back_img20[...,1]=255; back_img20[...,2]=155;
'''


#initialize network
fo=glob.glob(model_main_dir + 'netG_epoch_*')
model_name1=fo[0]
netM=ResnetConditionHR(input_nc=(3,3,1,4),output_nc=4,n_blocks1=7,n_blocks2=3)
netM=nn.DataParallel(netM)
netM.load_state_dict(torch.load(model_name1))
netM.cuda()
netM.eval()
cudnn.benchmark=True
reso=(1080,1080) #input reoslution to the network

print('Init Kinect...')
# Initialize the library with the path containing the module
pyK4A = pyKinectAzure(modulePath)

# Open device
pyK4A.device_open()

# Modify camera configuration
device_config = pyK4A.config
device_config.color_resolution = _k4a.K4A_COLOR_RESOLUTION_1080P
device_config.depth_mode = _k4a.K4A_DEPTH_MODE_NFOV_UNBINNED
print(device_config)

# Start cameras using modified configuration
pyK4A.device_start_cameras(device_config)
# Initialize the body tracker
pyK4A.bodyTracker_start(bodyTrackingModulePath)

# capture background image
sec = input('Ready?.\n')

# Get capture
pyK4A.device_get_capture()

# Get the color image from the capture
color_image_handle = pyK4A.capture_get_color_image()
# Read and convert the image data to numpy array:
bg_im0 = pyK4A.image_convert_to_numpy(color_image_handle)
# BGR to RGB
bg_im0=cv2.cvtColor(bg_im0,cv2.COLOR_BGR2RGB)

pyK4A.capture_release()

'''
#Create a list of test images
test_imgs = [f for f in os.listdir(data_path) if
			   os.path.isfile(os.path.join(data_path, f)) and f.endswith('_img.png')]
test_imgs.sort()

#output directory
result_path=args.output_dir
if not os.path.exists(result_path):
	os.makedirs(result_path)
'''

while True:
	#filename = test_imgs[i]	
	#original image

	# Get capture
	pyK4A.device_get_capture()

	# Get the depth image from the capture
	depth_image_handle = pyK4A.capture_get_depth_image()
	# Get the color image from the capture
	color_image_handle = pyK4A.capture_get_color_image()
	k=0

	if depth_image_handle and color_image_handle:

		bgr_img = pyK4A.image_convert_to_numpy(color_image_handle)
		bgr_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB)

		'''
		if args.back is None:
			#captured background image
			bg_im0 = cv2.imread(os.path.join(data_path, filename.replace('_img','_back')))
			bg_im0 = cv2.cvtColor(bg_im0,cv2.COLOR_BGR2RGB)
		'''

		#segmentation mask
		# Perform body detection
		pyK4A.bodyTracker_update()

		'''
		# Get the information of each body
		for body in pyK4A.body_tracker.bodiesNow:
			pyK4A.body_tracker.printBodyPosition(body)
		'''
		if pyK4A.get_num_bodies() > 0:
			# Read and convert the Depth image data to numpy array:
			depth_image = pyK4A.image_convert_to_numpy(depth_image_handle)

			# Get body segmentation image
			body_image_handle = pyK4A.bodyTracker_get_body_segmentation()
			transformed_custom_image = pyK4A.transform_depth_to_color_custom(depth_image_handle,body_image_handle, color_image_handle)

			rcnn = depth_to_mask(transformed_custom_image)

			'''
			if args.video: #if video mode, load target background frames
				#target background path
				back_img10=cv2.imread(os.path.join(args.target_back,filename.replace('_img.png','.png'))); back_img10=cv2.cvtColor(back_img10,cv2.COLOR_BGR2RGB);
				#Green-screen background
				back_img20=np.zeros(back_img10.shape); back_img20[...,0]=120; back_img20[...,1]=255; back_img20[...,2]=155;

				#create multiple frames with adjoining frames
				gap=20
				multi_fr_w=np.zeros((bgr_img.shape[0],bgr_img.shape[1],4))
				idx=[i-2*gap,i-gap,i+gap,i+2*gap]
				for t in range(0,4):
					if idx[t]<0:
						idx[t]=len(test_imgs)+idx[t]
					elif idx[t]>=len(test_imgs):
						idx[t]=idx[t]-len(test_imgs)

					file_tmp=test_imgs[idx[t]]
					bgr_img_mul = cv2.imread(os.path.join(data_path, file_tmp));
					multi_fr_w[...,t]=cv2.cvtColor(bgr_img_mul,cv2.COLOR_BGR2GRAY);

			else:
			'''	

			# create the multi-frame
			multi_fr_w=np.zeros((bgr_img.shape[0],bgr_img.shape[1],4))
			multi_fr_w[...,0] = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2GRAY)
			multi_fr_w[...,1] = multi_fr_w[...,0]
			multi_fr_w[...,2] = multi_fr_w[...,0]
			multi_fr_w[...,3] = multi_fr_w[...,0]
			
			#crop tightly?
			bgr_img0 = bgr_img
			bbox = get_bbox(rcnn, R = bgr_img0.shape[0], C = bgr_img0.shape[1])

			crop_list = [bgr_img, bg_im0, rcnn, multi_fr_w]
			crop_list = crop_images(crop_list,reso,bbox)

			bgr_img = crop_list[0]
			bg_im=crop_list[1]
			rcnn=crop_list[2]
			#back_img1=crop_list[3]
			#back_img2=crop_list[4]
			multi_fr=crop_list[3]

			#process segmentation mask
			kernel_er = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
			kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
			#rcnn=rcnn.astype(np.float32)/255; rcnn[rcnn>0.2]=1
			K=25

			zero_id=np.nonzero(np.sum(rcnn,axis=1)==0)
			del_id=zero_id[0][zero_id[0]>250]
			if len(del_id)>0:
				del_id=[del_id[0]-2,del_id[0]-1,*del_id]
				rcnn=np.delete(rcnn,del_id,0)
			rcnn = cv2.copyMakeBorder( rcnn, 0, K + len(del_id), 0, 0, cv2.BORDER_REPLICATE)


			rcnn = cv2.erode(rcnn, kernel_er, iterations=10)
			rcnn = cv2.dilate(rcnn, kernel_dil, iterations=5)
			rcnn=cv2.GaussianBlur(rcnn.astype(np.float32),(31,31),0)
			rcnn=(255*rcnn).astype(np.uint8)
			rcnn=np.delete(rcnn, range(reso[0],reso[0]+K), 0)


			#convert to torch
			img=torch.from_numpy(bgr_img.transpose((2, 0, 1))).unsqueeze(0); img=2*img.float().div(255)-1
			bg=torch.from_numpy(bg_im.transpose((2, 0, 1))).unsqueeze(0); bg=2*bg.float().div(255)-1
			rcnn_al=torch.from_numpy(rcnn).unsqueeze(0).unsqueeze(0); rcnn_al=2*rcnn_al.float().div(255)-1
			multi_fr=torch.from_numpy(multi_fr.transpose((2, 0, 1))).unsqueeze(0); multi_fr=2*multi_fr.float().div(255)-1

			
			with torch.no_grad():
				img, bg, rcnn_al, multi_fr =Variable(img.cuda()),  Variable(bg.cuda()), Variable(rcnn_al.cuda()), Variable(multi_fr.cuda())
				input_im=torch.cat([img,bg,rcnn_al,multi_fr],dim=1)
				
				alpha_pred,fg_pred_tmp=netM(img,bg,rcnn_al,multi_fr)
				
				al_mask = (alpha_pred>0.95).type(torch.cuda.FloatTensor)

				# for regions with alpha>0.95, simply use the image as fg
				fg_pred = img*al_mask + fg_pred_tmp*(1-al_mask)

				alpha_out = to_image(alpha_pred[0,...]); 

				#refine alpha with connected component
				labels = label((alpha_out>0.05).astype(int))

				try:
					assert( labels.max() != 0 )
				except:
					pass  # or you could use 'continue'

				largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
				alpha_out = alpha_out*largestCC

				alpha_out = (255*alpha_out[...,0]).astype(np.uint8)				

				fg_out = to_image(fg_pred[0,...])
				fg_out = fg_out*np.expand_dims((alpha_out.astype(float)/255>0.01).astype(float),axis=2); fg_out=(255*fg_out).astype(np.uint8)

				#Uncrop
				R0 = bgr_img0.shape[0];C0=bgr_img0.shape[1]
				#alpha_out0=uncrop(alpha_out,bbox,R0,C0)
				fg_out0 = uncrop(fg_out,bbox,R0,C0)

			#compose
			#back_img10=cv2.resize(back_img10,(C0,R0)); back_img20=cv2.resize(back_img20,(C0,R0))
			#comp_im_tr1=composite4(fg_out0,back_img10,alpha_out0)
			#comp_im_tr2=composite4(fg_out0,back_img20,alpha_out0)

			fg_out0 = cv2.cvtColor(fg_out0,cv2.COLOR_RGB2BGR)

			cv2.namedWindow('Background Matting Kinect',cv2.WINDOW_NORMAL)
			cv2.imshow('Background Matting Kinect', fg_out0)
			k = cv2.waitKey(1)

	# Release the image
	pyK4A.image_release(depth_image_handle)
	#pyK4A.image_release(body_image_handle)
	pyK4A.image_release(color_image_handle)
	pyK4A.image_release(pyK4A.body_tracker.segmented_body_img)

	pyK4A.capture_release()
	pyK4A.body_tracker.release_frame()

	# Esc key to stop
	if k==27:
		break
pyK4A.device_stop_cameras()
pyK4A.device_close()
#cv2.imwrite(result_path+'/'+filename.replace('_img','_out'), alpha_out0)
#cv2.imwrite(result_path+'/'+filename.replace('_img','_fg'), cv2.cvtColor(fg_out0,cv2.COLOR_BGR2RGB))
#cv2.imwrite(result_path+'/'+filename.replace('_img','_compose'), cv2.cvtColor(comp_im_tr1,cv2.COLOR_BGR2RGB))
#cv2.imwrite(result_path+'/'+filename.replace('_img','_matte').format(i), cv2.cvtColor(comp_im_tr2,cv2.COLOR_BGR2RGB))

#print('Done: ' + str(i+1) + '/' + str(len(test_imgs)))