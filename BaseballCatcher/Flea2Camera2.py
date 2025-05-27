### Wrapper for Baseball Catcher Flea 2 cameras

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
        self.ON = False # The cameras will be turned on by default
        self.camera1 = None
        self.camera2 = None
        self.frame_shape = (0,0,0)
        self.frame_rate = 0

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

        # disconnect camera
        self.camera1.disconnect()
        self.camera2.disconnect()


    ################## Camera Setup Functions

    """ Connect to the first availible camera
    """
    def connect(self):
        bus = pycap2.BusManager()
        self.camera1 = pycap2.Camera()
        self.camera2 = pycap2.Camera()
        uid1 = bus.getCameraFromIndex(0)
        uid2 = bus.getCameraFromIndex(1)
        self.camera1.connect(uid1)
        self.camera2.connect(uid2)

    """ Setup the camera to the Baseball Catcher Lab parameters
    """
    def setup(self):
        self.frame_shape = (480,640,3) # Opencv frames read row column  480x640
        self.frame_rate = 60

        # Setup Visual Inspection Mode
        self.setupBaseballCatcher(self.camera1)
        self.setupBaseballCatcher(self.camera2)

        # # Wait for setup to finish
        # time.sleep(1)

        # For now just go ahead and start
        self.start()

        # Wait for Camera to start
        # time.sleep(1)

    """ Setup specific to the Baseball Catcher lab
    """
    def setupBaseballCatcher(self,camera):
        ''' Flea2 parameters for VisualCapture
                Sutter_Speed = 12
                WhiteBalance_R = 560
                WhiteBalance_B = 740
                Gain_A = 200
                Gain_B = 0
                FPS = 60
                VideoMode = 640x480Y8
                Max Buffer 4
                Trigger On = 0
        '''
        # Meta Parameters
        SUTTER_SPEED = 2.0
        WHITE_BALANCE_R = 560
        WHITE_BALANCE_B = 740
        GAIN_A = 500
        GAIN_B = 300
        TRIGGER_ON = 0
        MAX_BUFFERS = 4

        # Make Black and White
        camera.setVideoModeAndFrameRate(pycap2.VIDEO_MODE.VM_640x480Y8 \
                                            ,pycap2.FRAMERATE.FR_60 )
        ## See if this is needed
        # camera.enableLUT(False)

        config = camera.getConfiguration()
        config.grabMode = pycap2.GRAB_MODE.DROP_FRAMES
        # camera.setConfiguration(grabMode = pycap2.GRAB_MODE.DROP_FRAMES)
        # config.asyncBusSpeed = pycap2.BUS_SPEED.S800
        config.isochBusSpeed = pycap2.BUS_SPEED.S800
        # camera.setConfiguration(isochBusSpeed = pycap2.BUS_SPEED.S800)
        # camera.setConfiguration(numBuffers = MAX_BUFFERS)
        # camera.setConfiguration(grabTimeOut = 5000)
        config.numBuffers = MAX_BUFFERS
        config.grabTimeOut = -1  # infinate
        camera.setConfiguration(config,False)

        # Set trigger state
        triggermode = pycap2.TriggerMode()
        triggermode.mode = 0
        triggermode.onOff = TRIGGER_ON
        triggermode.polarity = 0
        triggermode.source = 0
        triggermode.parameter = 0
        camera.setTriggerMode(triggermode,False)

        # Shutter
        shutter = camera.getProperty(pycap2.PROPERTY_TYPE.SHUTTER)
        shutter.onePush = False
        shutter.autoManualMode = False
        shutter.absControl = True
        shutter.onOff = True
        shutter.absValue = SUTTER_SPEED
        camera.setProperty(shutter,False)

        # White Balance
        # whiteB = camera.getProperty(pycap2.PROPERTY_TYPE.WHITE_BALANCE)
        # whiteB.absControl = False
        # whiteB.autoManualMode = False
        # whiteB.onOff = True
        # whiteB.valueA = WHITE_BALANCE_R
        # whiteB.valueB = WHITE_BALANCE_B
        # camera.setProperty(whiteB,True)

        # Gamma
        gamma = camera.getProperty(pycap2.PROPERTY_TYPE.GAMMA)
        gamma.absValue = 1.0
        camera.setProperty(gamma,False)
        
        # Sharpness
        sharp = camera.getProperty(pycap2.PROPERTY_TYPE.SHARPNESS)
        sharp.absControl = False
        sharp.valueA = 2000
        camera.setProperty(sharp,False)

        # Gain
        gain = camera.getProperty(pycap2.PROPERTY_TYPE.GAIN)
        gain.absControl = False
        gain.autoManualMode = False
        gain.onOff = True
        gain.valueA = GAIN_A
        gain.valueB = GAIN_B
        camera.setProperty(gain,False)

        

    """ Start capturing frames
    """
    def start(self):
        if not self.ON:
            self.camera1.startCapture()
            self.camera2.startCapture()
            # pycap2.startSyncCapture([self.camera1,self.camera2])
            self.ON = True

    """ Stop capturing frames
    """
    def stop(self):
        if self.ON ==  True:
            self.camera1.stopCapture()
            self.camera2.stopCapture()
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
        image1 = self.camera1.retrieveBuffer()
        image2 = self.camera2.retrieveBuffer()


        # image.convert(pycap2.PIXEL_FORMAT.BGR) # This didn't work
        # Convert image to np array 
        # assume setup of VM_640x480 RGB
        # cv_image1 = np.array(image1.getData(), dtype="uint8").reshape((image1.getRows(), image1.getCols(),3) )
        cv_image1 = np.array(image1.getData(), dtype="uint8").reshape((image1.getRows(), image1.getCols(),1) )
        cv_image1 = cv2.cvtColor(cv_image1, cv2.COLOR_RGB2BGR)  # Use Opencv to convert to BGR

        # cv_image2 = np.array(image2.getData(), dtype="uint8").reshape((image2.getRows(), image2.getCols(),3) )
        cv_image2 = np.array(image2.getData(), dtype="uint8").reshape((image2.getRows(), image2.getCols(),1) )
        cv_image2 = cv2.cvtColor(cv_image2, cv2.COLOR_RGB2BGR)  # Use Opencv to convert to BGR
        return cv_image1, cv_image2
        
