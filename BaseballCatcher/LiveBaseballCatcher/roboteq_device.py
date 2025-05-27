# This is a copy of a simple implementation of Roboteq devices core functions
# It is kind of inefficient and redundant 

# This code will need to be tested, it might not work 

# I only use the connect, send_command, and getdata functions

import serial
import time

# set serial read time out to 1 second (Some commands might take longer)
SERIAL_TIMEOUT = 1

class RoboteqDevice:
    """Initializes A Roboteq device object
    
    """
    def __init__(self):
        """Initializes A Roboteq device object
        
        """
        self.is_roboteq_connected = False
        self.port = ""
        self.baudrate = 115200
        self.ser = None

    """ Destructor 
    """
    def __del__(self):
        if self.is_roboteq_connected:
            if self.ser.isOpen():
                print("Disconnecting Roboteq")
                self.ser.close()

    def connect(self, port: str, baudrate: int = 115200):
        """Attempts to make a serial connection to the roboteq controller
        """
        self.port = port
        self.baudrate = baudrate
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=SERIAL_TIMEOUT
            )
            # reset the connection if need be
            if self.ser.isOpen():
                self.ser.close()
            self.ser.open()
            self.ser.flushInput()
            self.is_roboteq_connected = True

        except:
            print("ROBOTEQ ERROR: Unable to connect to roboteq motor controller")
            print("Please turn on the roboteq power.")
            self.is_roboteq_connected = False

        return self.is_roboteq_connected

    def set_config(
        self, config_item: str, channel: int = None, value: int = None
    ):
        self.send_command(f"{config_item} {channel} {value}")

    def get_value(
        self, config_item: str, channel: int = None, value: int = None
    ):
        self.getdata()
        self.send_command(f"{config_item} {channel} {value}")
        result = self.getdata()
        return result

    def getdata(self):
        # info = bytearray()
        info = self.ser.read(100)
        # info += self.ser.read(self.ser.inWaiting())
        # while self.ser.inWaiting() > 0:
        #     # info += str(self.ser.read())
        #     info += self.ser.read()

        return info.decode()

    def command_motor(
        self, command_item: str, channel: [str, int] = None, value: [str, int] = None
    ):
        """Sends a command to a specific motor channel
        
        """
        self.send_command(f"{command_item} {channel} {value}")

    def send_command(self, command: str):
        """Sends a string as a command

        Args:
            command (str): String to send as command

        """
        if command[-1] != "\r":
            command += "\r"
            self.ser.write(command.encode())
        else:
            self.ser.write(command.encode())
        # wait a little so command goes through
        # time.sleep(10/1000)
        # print(command)
