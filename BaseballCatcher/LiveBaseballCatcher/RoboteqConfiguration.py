import threading
import time
from tkinter import *
from tkinter import font
from tkinter import filedialog
from tkinter import messagebox
from roboteq_wrapper import RoboteqWrapper
from roboteq_constants import *

# Use $ ls /dev/tty* to find Keyspan port name
PORT = '/dev/ttyUSB0'                    # Linux
#PORT = '/dev/tty.serial-10007361C'      # Mac

# Button Definitions
SPEED = 0
POSITION = 1

class App(Frame):
    def __init__(self, winname='Move Catcher'):  # GUI Design
        self.root = Tk()

        helv18 = font.Font(family='Helvetica', size=18, weight='bold')
        # print("Width",windowWidth,"Height",windowHeight)
        self.root.wm_title(winname)
        positionRight = int(self.root.winfo_screenwidth() / 2 - 1024 / 2)
        positionDown = int(self.root.winfo_screenheight() / 2 - 768 / 2)
        # Positions the window in the center of the page.
        self.root.geometry("+{}+{}".format(positionRight, positionDown))
        self.root.wm_protocol("WM_DELETE_WINDOW", self.exitApp)
        display = Frame(self.root)

        # catcher control buttons
        Button(display, text="Reset Catcher", width=12, height=1, command=self.resetCatcher, font=helv18).grid(
            column=0, row=0)
        Button(display, text="Move Home", width=12, height=1, command=self.moveHome, font=helv18).grid(column=1, row=0)
        Button(display, text="Set Home", width=12, height=1, command=self.setHome, font=helv18).grid(column=1, row=1)
        # Radio Buttons
        self.motorMode = IntVar()
        Radiobutton(display, text="Speed", variable=self.motorMode, value=SPEED, width=8, font=helv18).grid(column=2, row=0, sticky=W)
        Radiobutton(display, text="Position", variable=self.motorMode, value=POSITION, width=10, font=helv18).grid(column=2, row=1, sticky=W)
        self.motorMode.set(SPEED)
        # sliders
        self.sliderSpeedX = DoubleVar()
        self.sliderSpeedY = DoubleVar()
        Label(display, text='X Motor', width=7, font=helv18).grid(column=3, row=0, sticky=E)
        Scale(display, from_=-1000, to=1000, resolution=50, length=255, width=10, variable=self.sliderSpeedX, font=helv18,
              orient='horizontal').grid(column=4, row=0)
        self.sliderSpeedX.set(0)
        Label(display, text='Y Motor', width=7, font=helv18).grid(column=3, row=1, sticky=E)
        Scale(display, from_=-1000, to=1000, resolution=50, length=255, width=10, variable=self.sliderSpeedY, font=helv18,
              orient='horizontal').grid(column=4, row=1)
        self.sliderSpeedY.set(0)
        self.sliderPositionX = DoubleVar()
        self.sliderPositionY = DoubleVar()
        Scale(display, from_=-CATCHER_W/2, to=CATCHER_W/2, resolution=0.1, length=255, width=10, variable=self.sliderPositionX, font=helv18,
              orient='horizontal').grid(column=5, row=0)
        self.sliderPositionX.set(0)
        Scale(display, from_=-CATCHER_H/2, to=CATCHER_H/2, resolution=0.1, length=255, width=10, variable=self.sliderPositionY, font=helv18,
              orient='horizontal').grid(column=5, row=1)
        self.sliderPositionY.set(0)
        # Encoder Count
        self.currentX = StringVar()
        Label(display, width=5, textvariable=self.currentX, font=helv18).grid(column=6, row=0, sticky=W)
        self.currentY = StringVar()
        Label(display, width=5, textvariable=self.currentY, font=helv18).grid(column=6, row=1, sticky=W)
        Button(display, text="Move Catcher", width=12, height=1, command=self.moveCatcher, font=helv18).grid(column=7, row=0)
        Button(display, text="Stop Catcher", width=12, height=1, command=self.stopCatcher, font=helv18).grid(column=7, row=1)
        Button(display, text="Quit", width=4, height=2, command=self.quitProgram, font=helv18).grid(column=8, row=0, rowspan=2)
        display.pack(side='right', ipadx=10)

        self.Catcher = RoboteqWrapper(PORT)
        if self.Catcher.Device.is_roboteq_connected:
            self.currentX.set(self.Catcher.GetEncoderCount(X_MOTOR))
            self.currentY.set(self.Catcher.GetEncoderCount(Y_MOTOR))
        else:
            self.currentX.set(0)
            self.currentY.set(0)
                
    def resetCatcher(self):
        self.Catcher.SetToDefault()

    def moveHome(self):
        self.Catcher.MoveAtSpeed(X_HOME_SPEED, Y_HOME_SPEED)

    def setHome(self):
        self.Catcher.setHome()
        self.currentX.set(self.Catcher.GetEncoderCount(X_MOTOR))
        self.currentY.set(self.Catcher.GetEncoderCount(Y_MOTOR))

    def moveCatcher(self):
        if self.motorMode.get() == SPEED:
            self.Catcher.MoveAtSpeed(self.sliderSpeedX.get(), self.sliderSpeedY.get())
        else:
            self.Catcher.MoveToXY(self.sliderPositionX.get(), self.sliderPositionY.get())

    def stopCatcher(self):
        self.Catcher.MoveAtSpeed(0, 0)
        self.currentX.set(self.Catcher.GetEncoderCount(X_MOTOR))
        self.currentY.set(self.Catcher.GetEncoderCount(Y_MOTOR))

    def run(self):  # run main loop
        self.root.mainloop()

    def quitProgram(self):  # click on the Quit button
        t = threading.Timer(1, self.stop)  # start a timer (non-blocking) to give main thread time to stop
        t.start()

    def exitApp(self):  # click on the red cross (quit) button
        t = threading.Timer(1, self.stop)  # start a timer (non-blocking) to give main thread time to stop
        t.start()

    def stop(self):
        del self.Catcher
        self.root.quit()


app = App()
app.run()
