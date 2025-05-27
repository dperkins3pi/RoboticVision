# This is our custom wrapper 
#  that mimics the functionality of Prof Lee's wrapper in c++
# It is mostly Full of write commands with a few parameters
import threading
from roboteq_device import RoboteqDevice
from roboteq_constants import *
import numpy as np
import time

class RoboteqWrapper:
    """ Custom Roboteq device wrapper
        Initializes Roboteq device and setups up config
    """
    def __init__(self, Port):
        # Create a Robot motor device object and connect it
        self.Device = RoboteqDevice()
        self.Device.connect(Port)
        # ^ECHOF nn   nn:  0 eanbled 1 disabled
        command = "^ECHOF {}\r".format(1)
        self.Device.send_command(command) # 1: disabled, 0:enabled
        # Flush the I/O by asking for version info (if connected)
        if self.Device.is_roboteq_connected:
            self.Device.send_command("?FID \r".format())    # Board UID
            time.sleep(1)
            data = self.Device.getdata()
            print(data)             # Print UID to show availability of the board
            if len(data) > 10:
                self.Device.is_roboteq_connected = True
                # Set to open loop speed mode and set speed to 0
                self.MoveAtSpeed(0, 0)
            else:
                print("Catcher is not available!.")
                self.Device.is_roboteq_connected = False

        else:
            print("Catcher is not connected!")
            self.Device.is_roboteq_connected = False

        """Use DIN3 and DIN4 for safety stop"""
        # Set DIN3 nd DIN4 to be active low
        # L1 + (L2 * 2) + (L3 * 4) + (L4 * 8) + (L5 * 16) + (L6 * 32) 1 for Active Low or 0 for Active High
        command = "^DINL {}\r".format(63, 0)  # Set all to active high
        self.Device.send_command(command)     # This value will set din to rnu script
        for switches in range(1, 6):
            # ^DINA cc (aa + mm) cc:Channel number, aa = 1 : safety stop,  mm = mot1 * 16 + mot2 * 32
            self.Device.send_command("^DINA {} {}\r".format(switches, X_OFFSET + 0))  # Set all to no action as default

        # ^DINA cc (aa + mm) cc:Channel number, aa = 0: no action 1 : safety stop,  mm = mot1 * 16 + mot2 * 32
        command = "^DINA {} {}\r".format(X_SWITCH, X_OFFSET + 0)
        self.Device.send_command(command)  # This value will set din to rnu script
        # ^DINA cc (aa + mm) cc:Channel number, aa = 0: no action 1 : safety stop,  mm = mot1 * 16 + mot2 * 32
        command = "^DINA {} {}\r".format(Y_SWITCH, Y_OFFSET + 0)
        self.Device.send_command(command)  # This value will set din to rnu script
        # Run intiallization functions

    """ Load default values from constants definitions
    """
    def init_variables(self):
        if self.Device.is_roboteq_connected: # check if connected
            self.Device.command_motor('^MXPF', X_MOTOR, INIT_POWER)
            self.Device.command_motor('^MXPF', Y_MOTOR, INIT_POWER)
            self.Device.command_motor('^MXPR', X_MOTOR, INIT_POWER)
            self.Device.command_motor('^MXPR', Y_MOTOR, INIT_POWER)
            self.SetWDT(WDT_TIMEOUT)

    """ Destructor 
    """
    def __del__(self):
        self.MoveAtSpeed(0, 0)      # XSpeed and YSpeed
        self.SaveEEPROM()           # This is probably needed to clear out the read buffer
        del self.Device

    """ Set Device Default 
            (Only needed if motor control's config is unexpectedly lost)
    """
    def ResetToFactory(self):       # Reset EEPROM to Factory
        key = 321654987
        command = "%EERST {}\r".format(key)
        self.Device.send_command(command)  # set encoder operation mode to feedback. value = MOT1 (16) + FEEDBACK (2)
        
    def SetToDefault(self):
        if not self.Device.is_roboteq_connected:
            return
        # LoadEEPROM(i);	#Actually there is no need to load from EEPROM to RAM because it is done at power up automatically
        # Setting indivisual motors
        motors = [Y_MOTOR, X_MOTOR]
        for motor in motors:
            # ^EMOD cc (aa + mm) cc:Channel number, aa = 0 : Unused 1 : Command 2 : Feedback, mm = mot1 * 16 + mot2 * 32
            offset = Y_OFFSET + 2 if (motor == Y_MOTOR) else X_OFFSET + 2
            command = "^EMOD {} {}\r".format(motor, offset)
            self.Device.send_command(command)  # set encoder operation mode to feedback. value = MOT1 (16) + FEEDBACK (2)

            # ^ELL cc nn     cc : Channel number, nn : low count limit (default -20000)
            command = "^ELL {} {}\r".format(motor, 0)
            self.Device.send_command(command)  # set encoder low limit

            # ^EHL cc nn     cc : Channel number, nn : high count limit (default 20000)
            highCount = Y_INCH_COUNTS*CATCHER_H if (motor == Y_MOTOR) else X_INCH_COUNTS*CATCHER_W
            command = "^EHL {} {}\r".format(motor, highCount)
            self.Device.send_command(command)  # set encoder high limit

            # Set encoder home value.  This value will be loaded in the selected encoder counter when a home switch is detected,
            # or when a Home command is received from the serial / USB, or issued from a MicroBasic script.
            # Y home value is zero.  X home value is the HighCount because encoder signals A and B are swapped.

            # ^EHOME cc nn     cc : Channel number, nn : home value
            homevalue = 0 if (motor == Y_MOTOR) else 0   #X_INCH_COUNTS*CATCHER_W
            command = "^EHOME {} {}\r".format(motor, homevalue)
            self.Device.send_command(command)  # This value will be loaded when the stage moves to the home position

            # ^EPPR cc nn   cc : Channel number, nn : PPR value (1 to 5000)
            ppr = Y_MOTOR_PPR if (motor == Y_MOTOR) else X_MOTOR_PPR
            command = "^EPPR {} {}\r".format(motor, ppr)
            self.Device.send_command(command)  # set encoder pulse per revolution value

            # Close Loop Parameters
            # ^MVEL cc nn   cc: channel, nn:  velocity in RPM
            VEL = Y_POS_VEL if (motor == Y_MOTOR) else X_POS_VEL
            command = "^MVEL {} {}\r".format(motor, VEL)
            self.Device.send_command(command)  # Velocity in RPM in position mode

            # ^MVEL cc nn   cc: channel, nn:  velocity in RPM
            TURN = Y_TURN_MM if (motor == Y_MOTOR) else X_TURN_MM
            command = "^MXTRN {} {}\r".format(motor, TURN)
            self.Device.send_command(command)  # Maximum number of turns between limits

            # ^ICAP cc nn   cc: channel, nn: Integral cap in % (1% to 100%)
            ICAP = Y_PID_ICAP if (motor == Y_MOTOR) else X_PID_ICAP
            command = "^ICAP {} {}\r".format(motor, ICAP)
            self.Device.send_command(command)  # Default 100%

            # ^KP cc nn   cc: channel, nn:  Gain *10 (0  to 250)
            KP = Y_PID_KP if (motor == Y_MOTOR) else X_PID_KP
            command = "^KP {} {}\r".format(motor, KP)
            self.Device.send_command(command)

            # ^KI cc nn   cc: channel, nn:  Deceleration time in 0.1 RPM per second (100 to 32000)
            KI = Y_PID_KI if (motor == Y_MOTOR) else X_PID_KI
            command = "^KI {} {}\r".format(motor, KI)
            self.Device.send_command(command)

            # ^KD cc nn   cc: channel, nn:  Deceleration time in 0.1 RPM per second (100 to 32000)
            KD = Y_PID_KD if motor == Y_MOTOR else X_PID_KD
            command = "^KD {} {}\r".format(motor, KD)
            self.Device.send_command(command)

            # Set General Power Settings that are the same for both motors
            # ^ALIM cc nn     cc : Channel number, nn :  amp limit = amps * 10, e.g., enter 200 for 20 amps
            command = "^ALIM {} {}\r".format(motor, AMPS_LIMIT)
            self.Device.send_command(command)  # Input value = amps * 10, e.g., enter 200 for 20 amps

            # Amps Triger 20 A, Amps Trigger Delay 500 mSec - Not needed if no action is required
            # ^MXPF cc nn (25~100)
            command = "^MXPF {} {}\r".format(motor, INIT_POWER)
            self.Device.send_command(command)  # Input value = %  Set it to 25% first and adjust later if necessary

            # ^MXPR cc nn (25~100)
            command = "^MXPR {} {}\r".format(motor, INIT_POWER)
            self.Device.send_command(command)  # Input value = %  Set it to 25% first and adjust later if necessary

            # Motor Settings
            # ^MXRPM cc nn   cc: channel, nn: Max RPM (10 to 65000)
            command = "^MXRPM {} {}\r".format(motor, MAX_RPM)
            self.Device.send_command(command)  # Maximum RPM that is to be set as +1,000 for relative speed

            # ^MAC cc nn   cc: channel, nn:  Acceleration time in 0.1 RPM per second (100 to 32000)
            command = "^MAC {} {}\r".format(motor, ACCELERATION)
            self.Device.send_command(
                command)  # Aceleration self.Catcher.setHome()is 0.1RPM per second.  1,000 will acelerate 100RPM per esecond

            # ^MDEC cc nn   cc: channel, nn:  Deceleration time in 0.1 RPM per second (100 to 32000)
            command = "^MDEC {} {}\r".format(motor, DECELERATION)
            self.Device.send_command(
                command)  # Deceleration is 0.1RPM per second.  1,000 will decelerate 100RPM per esecond

             # ^CLERD cc nn   cc: channel, nn:  nn = 0 : Detection disabled, 1 : 250ms at Error > 100, 2 : 500ms at Error > 250, 3 : 1000ms at Error > 500 (default 2)
            command = "^CLERD {} {}\r".format(motor, 3)
            self.Device.send_command(
                command)  # 0:Detection disabled 1:250ms at Error > 10SetMotorMode0 2:500ms at Error > 250  3:1000ms at Error > 500

        # Set General Power Settings for the board
        # ^PWMF cc nn   nn:  Frequency * 10 (10 to 200 or 1 to 20 KHz, default 180=18.0 KHz)
        command = "^PWMF {}\r".format(PWM_FREQ)
        self.Device.send_command(command) # Input value = Frequency in KHz * 10, e.g., enter 200 for 20 KHz			
	
        # ^OVL nn   nn:  voltage * 10, e.g., enter 350 for 35 volts
        command = "^OVL {}\r".format(OVER_VOLTAGE)
        self.Device.send_command(command) # Input value = voltage * 10, e.g., enter 350 for 35 volts			

        # ^UVL nn   nn:  voltage * 10, e.g., enter 50 for 5 volts
        command = "^UVL {}\r".format(UNDER_VOLTAGE)
        self.Device.send_command(command) # Input value = voltage * 10, e.g., enter 50 for 5 volts							

        self.SaveEEPROM()

    """ Sets Motor Mode
    """
    def SetMotorMode(self, channel, mode):
        if not self.Device.is_roboteq_connected:
            return
        command = "^MMOD {} {}".format(channel, mode)
        self.Device.send_command(command)

    """ Sets Encoder Values
    """
    def SetEncoderCount(self, channel, count):
        if not self.Device.is_roboteq_connected:
            return
        # Threshold max and min count limits
        count = 0 if (count < 0) else count
        highCount = Y_INCH_COUNTS*CATCHER_H if channel == Y_MOTOR else X_INCH_COUNTS*CATCHER_W
        count = highCount if (count > highCount) else count
        #Set Encoder Counters  !C [nn] mm for the current location
        command = "!C {} {}\r".format(channel, int(count))
        self.Device.send_command(command)

    def GetEncoderCount(self, channel):
        if not self.Device.is_roboteq_connected:
            return

        # Set Encoder Counters  !C [nn] mm for the current location
        command = "?C {}\r".format(channel)
        self.Device.send_command(command)
        data = self.Device.getdata()
        #print(data)
        start = data.find("C=")
        if start == -1:  # invalid response
            print("ERROR: Motor Response missed")
            return -1           #TODO: I think this happens when the device takes a long time to respond
        else:
            start = start + 2   # add two characters to start indexing after equals

        end = len(data) -1      # stop indexing 1 character from end of length (ignore "\r")
        if data[start:end].isdecimal():
            return int(data[start:end])
        else:
            return data[start:end]

    def SetEncoderHome(self, channel):
        if not self.Device.is_roboteq_connected:
            return
        # !H [nn] Set Encoder Counter to home value that is already set with EHOME command
        command = "!H {}\r".format(channel)
        self.Device.send_command(command)

    def GetMotorSpeed(self, channel):
        if not self.Device.is_roboteq_connected:
            return

	    # Read motor speed in RPM
        command = "?S {}\r".format(channel)
        self.Device.send_command(command)
        time.sleep(1)
        data = self.Device.getdata()
        print(data)
        start = data.find("S=")
        if start == -1:  # invalid response 
            print("ERROR: Motor Response missed")
            return -1               #TODO: I think this happens when the device takes a long time to respond
        else:
            start = start + 2       # add two characters to start indexing after equals

        end = len(data) -1          # stop indexing 1 character from end of length (ignore "\r")
        if data[start:end].isdecimal():
            return int(data[start:end])
        else:
            return data[start:end]

        """ Check if motor is stopped. 
            --If not, wait for motor to stop
            --It is redundant to call this in an if statement
    """
    def IsMotorStopped(self, channel):
        # if the devices is off assume motor can't be moving
        if not self.Device.is_roboteq_connected:
            return True
        # wait for motor to stop
        self.currentEncoder = self.GetEncoderCount(channel)
        time.sleep(TIMER)
        while self.currentEncoder != self.GetEncoderCount(channel):
            time.sleep(TIMER)
            self.currentEncoder = self.GetEncoderCount(channel)
        return True

###################################### Movement functions
    """ Set encoder home counts after mootors stop
    """
    def setHome(self):
        # Only if the device is connected
        if not self.Device.is_roboteq_connected:
            return
        self.SetEncoderHome(X_MOTOR)
        self.SetEncoderHome(Y_MOTOR)
        self.MoveAtSpeed(0, 0)

    """ Move the catcher back to center
        --Only if current motor positions are correct
    """
    def Center(self):
        self.SetMotorMode(X_MOTOR, CLOSED_LOOP_POS);
        self.SetMotorMode(Y_MOTOR, CLOSED_LOOP_POS);
        self.Move(0.0, 0.0)

    """ Set open loop sppeed -1000 ~ 1000 or close loop position
    """
    def MoveAtSpeed(self, XSpeed, YSpeed):
        if not self.Device.is_roboteq_connected:
            return 0
        # Set the movement to open loop
        self.SetMotorMode(X_MOTOR, OPEN_LOOP_SP)
        self.SetMotorMode(Y_MOTOR, OPEN_LOOP_SP)
        # !G [nn] mm, mm : -1000 ~ 1000 for single motor
        XSpeed = -MAX_RANGE_X if (XSpeed < -MAX_RANGE_X) else XSpeed
        XSpeed = MAX_RANGE_X if (XSpeed > MAX_RANGE_X) else XSpeed
        YSpeed = -MAX_RANGE_Y if (YSpeed < -MAX_RANGE_Y) else YSpeed
        YSpeed = MAX_RANGE_Y if (YSpeed > MAX_RANGE_Y) else YSpeed
        Speed = [0]*3       # Need 3 but the first one (0) is not used
        Speed[X_MOTOR] = XSpeed;
        Speed[Y_MOTOR] = YSpeed;
        command = "!$01 {} {}".format(int(Speed[1]), int(Speed[2]))     # First value must be for channel 1, and second value for channel 2
        #command = "!G {} {}\r".format(X_MOTOR, int(XSpeed))
        self.Device.send_command(command)
        return

    """ Send a Move command to the motors
    """
    def MoveToXY(self, x, y):
        # Device.Write(command) # !$01 %d %d
        if not self.Device.is_roboteq_connected:
            return
        # Set the movement to open loop
        self.SetMotorMode(X_MOTOR, CLOSED_LOOP_POS)
        self.SetMotorMode(Y_MOTOR, CLOSED_LOOP_POS)
        #self.Device.send_command("~MMOD {}\r".format(X_MOTOR))
        #print(self.Device.getdata())
        #self.Device.send_command("~MMOD {}\r".format(Y_MOTOR))
        #print(self.Device.getdata())
        y_relative = y * 2000 / CATCHER_H # Total distance travel is normalized to -1000 ~ +1000
        y_relative = MAX_RANGE_Y if (y_relative >= MAX_RANGE_Y) else y_relative
        y_relative = -MAX_RANGE_Y if (y_relative <= -MAX_RANGE_Y) else y_relative

        x_relative = x * 2000 / CATCHER_W # Total distance travel is normalized to -1000 ~ +1000
        x_relative = MAX_RANGE_X if (x_relative >= MAX_RANGE_X) else x_relative
        x_relative = -MAX_RANGE_X if (x_relative <= -MAX_RANGE_X) else x_relative
        Speed = [0]*3       # Need 3 but the first one (0) is not used
        Speed[X_MOTOR] = x_relative;
        Speed[Y_MOTOR] = y_relative;
        #print(Speed[X_MOTOR], Speed[Y_MOTOR])
        command = "!$01 {} {}".format(int(Speed[1]), int(Speed[2]))     # NOT ANYMORE-- 1: Y_MOTOR, 2: X_MOTOR
        self.Device.send_command(command)

    ###################################### System functions
    """ This asks the motor to send back some info
    """
    def SaveEEPROM(self):
        # Device.Write(Command) # "!G %d %d
        if not self.Device.is_roboteq_connected:
            return 0
        self.Device.send_command('%EESAV')  # need two %'s to specify one % (at least in c++)
        info = self.Device.getdata()
        print(info)

    """ set watchdog timer timeout value in mSec, 0 to disable.  
            command moves to 0 after the timeout
    """
    def SetWDT(self, duration):
        # Device.Write # ^RWD
        command = "^RWD {}".format(duration)
        self.Device.send_command(command)

