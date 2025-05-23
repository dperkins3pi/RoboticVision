"""
ECEn-631 Visual Inspection Project created by Harrison Garrett in 2020

"""

import PyCapture2 as pycap2
import cv2
import numpy as np
import time

### NEED TO REDO for Old Flea2 parameters
class FleaCam():

    ################### Init and Destruct functions

    """ List all local variables and Init the camera
    """
    def __init__(self):
        self.ON = False # The camera will be turned on by default
        self.camera = None
        self.frame_shape = (0,0,0)
        self.frame_rate = 0
        self.recorder = pycap2.FlyCapture2Video()
        self.recording = False
        # self.ImgFile = "" # TODO: Image recording Not implemented
        # self.ImgSave = False

        # Automatically connect and setup camera
        try:
            self.connect()
            self.setup()
        except Exception as e:
            print("ERROR: No camera found or invalid configuration")
            raise e

    """ Close camera and video recording
    """
    def __del__(self):
        # if the camera is on turn it off
        if self.ON:
            self.stop()

        # if recording close video file
        if self.recording:
            self.stopRecord()

        # disconnect camera
        self.camera.disconnect()


    ################## Camera Setup Functions

    """ Connect to the first availible camera
    """
    def connect(self):
        bus = pycap2.BusManager()
        self.camera = pycap2.Camera()
        uid = bus.getCameraFromIndex(0)
        self.camera.connect(uid)

    """ Setup the camera to the Visual Inspection Lab parameters
    """
    def setup(self):
        self.camera.setVideoModeAndFrameRate(pycap2.VIDEO_MODE.VM_640x480RGB \
                                            ,pycap2.FRAMERATE.FR_60 )
        self.frame_shape = (480,640,3) # Opencv frames read row column  480x640
        self.frame_rate = 60

        # Setup Visual Inspection Mode
        self.setupVisInspection()

        # # Wait for setup to finish
        # time.sleep(1)

        # For now just go ahead and start
        self.start()

        # Wait for Camera to start
        # time.sleep(1)

    """ Setup specific to the Visual Inspection lab
    """
    def setupVisInspection(self):
        ''' Flea2 parameters for VisualCapture
                Sutter_Speed = 12
                WhiteBalance_R = 560
                WhiteBalance_B = 740
                Gain_A = 200
                Gain_B = 0
                FPS = 60
                VideoMode = 640x480RGB
                Max Buffer 4
                Trigger On = 0
        '''
        # Meta Parameters
        SUTTER_SPEED = 12.0
        WHITE_BALANCE_R = 560
        WHITE_BALANCE_B = 740
        GAIN_A = 200
        GAIN_B = 0
        TRIGGER_ON = 0
        MAX_BUFFERS = 4

        ## See if this is needed
        self.camera.enableLUT(False)

        # config = self.camera.getConfiguration()
        # config.grabMode = pycap2.GRAB_MODE.DROP_FRAMES
        # config.asyncBusSpeed = pycap2.BUS_SPEED.S800
        # config.isochBusSpeed = pycap2.BUS_SPEED.S800
        # config.numBuffers = MAX_BUFFERS
        # # config.grabTimeOut = infinate ?
        # self.camera.setConfiguration(config,False)

        # Set trigger state
        triggermode = pycap2.TriggerMode()
        triggermode.mode = 0
        triggermode.onOff = TRIGGER_ON
        triggermode.polarity = 0
        triggermode.source = 0
        triggermode.parameter = 0
        self.camera.setTriggerMode(triggermode,True)

        # Shutter
        shutter = self.camera.getProperty(pycap2.PROPERTY_TYPE.SHUTTER)
        shutter.onePush = False
        shutter.autoManualMode = False
        shutter.absControl = True
        shutter.onOff = True
        shutter.absValue = SUTTER_SPEED
        self.camera.setProperty(shutter,True)

        # White Balance
        whiteB = self.camera.getProperty(pycap2.PROPERTY_TYPE.WHITE_BALANCE)
        whiteB.absControl = False
        whiteB.autoManualMode = False
        whiteB.onOff = True
        whiteB.valueA = WHITE_BALANCE_R
        whiteB.valueB = WHITE_BALANCE_B
        self.camera.setProperty(whiteB,True)

        # Gain
        gain = self.camera.getProperty(pycap2.PROPERTY_TYPE.GAIN)
        gain.absControl = False
        gain.autoManualMode = False
        gain.onOff = True
        gain.valueA = GAIN_A
        gain.valueB = GAIN_B
        self.camera.setProperty(gain,True)

        

    """ Start capturing frames
    """
    def start(self):
        if not self.ON:
            self.camera.startCapture()
            self.ON = True

    """ Stop capturing frames
    """
    def stop(self):
        if self.ON ==  True:
            self.camera.stopCapture()
            self.ON = False

    ################ Frame capturings

    """ Get a single frame from camera in Opencv format
            Converts from bytes to uint8,
            Saves image to uncompressed .avi file if recording
            Converts to BGR format (used by Opencv)
        Returns: numpy array, uint8 array in BGR 
    """
    def getFrame(self):
        # Get Bytes from Camera
        image = self.camera.retrieveBuffer()

        # save image if recording
        if self.recording:
            self.recorder.append(image)

        # if self.ImgSave: #TODO: Image recording Not implemented
        #     image.save()


        # image.convert(pycap2.PIXEL_FORMAT.BGR) # This didn't work
        # Convert image to np array 
        # assume setup of VM_640x480 RGB
        cv_image = np.array(image.getData(), dtype="uint8").reshape((image.getRows(), image.getCols(),3) )
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)  # Use Opencv to convert to BGR
        return cv_image


    ###################### Extra Video and Image capture functions using PyCapture2

    """ Init the video recording file
            To save frames call getFrame() at the specified frame rate
    """
    def initRecord(self,filename):
        # print(filename)
        self.recorder.AVIOpen(filename.encode(),self.frame_rate)
        self.recording = True

    """ Close the .avi video file
    """
    def stopRecord(self):
        self.recorder.close()
        self.recording = False

    """ 
    """
    # def initImgCap(self,filename):  #TODO: Image recording Not implemented
    #     # print(filename)
    #     self.ImgFile = filename
    #     self.ImgSave = True 
    
    # def stopImgCap(self):
    #     self.ImgSave = False

        
