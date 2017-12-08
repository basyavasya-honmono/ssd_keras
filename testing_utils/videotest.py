""" A class for testing a SSD model on a video file or webcam """

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image 
import pickle
import numpy as np
from random import shuffle
from scipy.misc import imread, imresize
from timeit import default_timer as timer

import sys
sys.path.append("..")
from ssd_utils import BBoxUtility


class VideoTest(object):
    """ Class for testing a trained SSD model on a video file and show the
        result in a window. Class is designed so that one VideoTest object 
        can be created for a model, and the same object can then be used on 
        multiple videos and webcams.
        
        Arguments:
            class_names: A list of strings, each containing the name of a class.
                         The first name should be that of the background class
                         which is not used.
                         
            model:       An SSD model. It should already be trained for 
                         images similar to the video to test on.
                         
            input_shape: The shape that the model expects for its input, 
                         as a tuple, for example (300, 300, 3)    
                         
            bbox_util:   An instance of the BBoxUtility class in ssd_utils.py
                         The BBoxUtility needs to be instantiated with 
                         the same number of classes as the length of        
                         class_names.
    
    """
    
    def __init__(self, class_names, model, input_shape):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = model
        self.input_shape = input_shape
        self.bbox_util = BBoxUtility(self.num_classes)
        
        # Create unique and somewhat visually distinguishable bright
        # colors for the different classes.
        self.class_colors = []
        for i in range(0, self.num_classes):
            # This can probably be written in a more elegant manner
            hue = 255*i/self.num_classes
            col = np.zeros((1,1,3)).astype("uint8")
            col[0][0][0] = hue
            col[0][0][1] = 128 # Saturation
            col[0][0][2] = 255 # Value
            cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
            col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
            self.class_colors.append(col) 
        
    def run(self, video_path = 0, start_frame = 0, conf_thresh = 0.6):
        """ Runs the test on a video (or webcam)
        
        # Arguments
        video_path: A file path to a video to be tested on. Can also be a number, 
                    in which case the webcam with the same number (i.e. 0) is 
                    used instead
                    
        start_frame: The number of the first frame of the video to be processed
                     by the network. 
                     
        conf_thresh: Threshold of confidence. Any boxes with lower confidence 
                     are not visualized.
                    
        """
    
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))
        
        # Compute aspect ratio of video     
        vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        vidar = vidw/vidh
        
        # Skip frames until reaching start_frame
        if start_frame > 0:
            vid.set(cv2.cv.CV_CAP_PROP_POS_MSEC, start_frame)
            
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        kaisuu=0
        oto=0
        dog=0
        car=0
        bicycle=0
        bus=0
        cat=0
        cow=0
        horse=0
        motorbike=0
        sheep=0
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (400,300))


    
        while True:
            retval, orig_image = vid.read()
            if not retval:
                print("Done!")
                return
            kaisuu=kaisuu+1
            if kaisuu % 1 !=0 :
               continue
             
            im_size = (self.input_shape[0], self.input_shape[1])    
            resized = cv2.resize(orig_image, im_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            # Reshape to original aspect ratio for later visualization
            # The resized version is used, to visualize what kind of resolution
            # the network has to work with.
            to_draw = cv2.resize(resized, (int(self.input_shape[0]*vidar), self.input_shape[1]))
            #†size Confirmation†
            print(to_draw.shape)
            # Use model to predict 
            inputs = [image.img_to_array(rgb)]
            tmp_inp = np.array(inputs)
            x = preprocess_input(tmp_inp)
            
            y = self.model.predict(x)
            
            
            # This line creates a new TensorFlow device every time. Is there a 
            # way to avoid that?
            results = self.bbox_util.detection_out(y)
            
            if len(results) > 0 and len(results[0]) > 0:
                # Interpret output, only one frame is used 
                det_label = results[0][:, 0]
                print("det:",det_label)
                det_conf = results[0][:, 1]
                det_xmin = results[0][:, 2]
                det_ymin = results[0][:, 3]
                det_xmax = results[0][:, 4]
                det_ymax = results[0][:, 5]

                top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                print("person:",15 in top_label_indices )
                print("dog:",12 in top_label_indices )
                print("car:",7 in top_label_indices )
                print("bicycle:",2 in top_label_indices )
                print("bus:",6 in top_label_indices )
                print("cat:",8 in top_label_indices )
                print("cow:",10 in top_label_indices )
                print("horse:",13 in top_label_indices )
                print("motorbike:",14 in top_label_indices )
                print("sheep:",17 in top_label_indices )
                print("top:",top_label_indices)
                
                #person beep
                if (15 in top_label_indices):
                    oto=oto+1
                    if oto==3 :
                       oto=0
                       print ('\007')
                else:
                    oto=0
                #dog beep
                if (12 in top_label_indices):
                    dog=dog+1
                    if dog==3 :
                       dog=0
                       print ('\007')
                else:
                    dog=0
                #car beep
                if (7 in top_label_indices):
                    car=car+1
                    if car==3 :
                       car=0
                       print ('\007')
                else:
                    car=0
                #bicycle beep
                if (2 in top_label_indices):
                    bicycle=bicycle+1
                    if bicycle==3 :
                       bicycle=0
                       print ('\007')
                else:
                    bicycle=0
                #bus beep
                if (6 in top_label_indices):
                    bus=bus+1
                    if bus==3 :
                       bus=0
                       print ('\007')
                else:
                    bus=0
                #cat beep
                if (8 in top_label_indices):
                    cat=cat+1
                    if cat==3 :
                       cat=0
                       print ('\007')
                else:
                    cat=0
                #cow beep
                if (10 in top_label_indices):
                    cow=cow+1
                    if cow==3 :
                       cow=0
                       print ('\007')
                else:
                    cow=0
                #horse beep
                if (13 in top_label_indices):
                    horse=horse+1
                    if horse==3 :
                       horse=0
                       print ('\007')
                else:
                    horse=0
                #motorbike beep
                if (14 in top_label_indices):
                    motorbike=motorbike+1
                    if motorbike==3 :
                       motorbike=0
                       print ('\007')
                else:
                    motorbike=0
                #sheep beep
                if (17 in top_label_indices):
                    sheep=sheep+1
                    if sheep==3 :
                       sheep=0
                       print ('\007')
                else:
                    sheep=0

                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                for i in range(top_conf.shape[0]):
                    xmin = int(round(top_xmin[i] * to_draw.shape[1]))
                    ymin = int(round(top_ymin[i] * to_draw.shape[0]))
                    xmax = int(round(top_xmax[i] * to_draw.shape[1]))
                    ymax = int(round(top_ymax[i] * to_draw.shape[0]))

                    # Draw the box on top of the to_draw image
                    class_num = int(top_label_indices[i])
                    cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax), 
                                  self.class_colors[class_num], 2)
                    text = self.class_names[class_num] + " " + ('%.2f' % top_conf[i])

                    text_top = (xmin, ymin-10)
                    text_bot = (xmin + 80, ymin + 5)
                    text_pos = (xmin + 5, ymin)
                    cv2.rectangle(to_draw, text_top, text_bot, self.class_colors[class_num], -1)
                    cv2.putText(to_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            
            # Calculate FPS
            # This computes FPS for everything, not just the model's execution 
            # which may or may not be what you want
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            
            # Draw FPS in top left corner
            cv2.rectangle(to_draw, (0,0), (50, 17), (255,255,255), -1)
            cv2.putText(to_draw, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)
            
            # write the flipped frame
            out.write(to_draw)

            cv2.imshow("SSD result", to_draw)
            if cv2.waitKey(1) & 0xFF == ord('q'):
               out.release() 
               break


            
        
