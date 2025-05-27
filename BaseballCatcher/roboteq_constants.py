# Alot of these are not used, They were copied from multiple sources
# Hard coded values are also used in the roboteq_wrapper.py

_AC = "!AC"
_AX = "!AX"
_B = "!B"
_BND = "!BND"
_C = "!C"
_CB = "!CB"
_CG = "!CG"
_CS = ""
_D0 = ""
_D1 = ""
_DC = ""
_DS = ""
_DX = ""
_EES = ""
_EX = ""
_G = "!G"
_H = ""
_MG = ""
_MS = ""
_P = ""
_PR = ""
_PRX = ""
_PX = ""
_R = ""
_RC = ""
_S = ""
_SX = ""
_VAR = ""

_DR = "?DR"


_MMOD = "^MMOD"
_RWD = "^RWD"
_MXMD = "^MXMD"
_MXRPM = "^MXRPM"

# Administration Functions
OPEN_LOOP_SP = 0		# open loop speed mode
CLOSED_LOOP_SP = 1		# closed loop speed
CLOSED_LOOP_POS = 2     # closed loop position relative
CLOSED_LOOP_CNT = 3     # closed loop count position

MAX_RANGE_X = 800		    # Closed loop position mode can only send relative position +/- 1000
MAX_RANGE_Y = 900		    # Closed loop position mode can only send relative position +/- 1000
TIMER = 0.1		        # timer interval
WDT_TIMEOUT = 0		    # WDT timeout in mSec, 0 to disable .  Delay for WDT_TIMEOUT seconds then set command to 0 position.

# Board Confiburations
X_MOTOR = 1		    # Channel 1 is used for X
Y_MOTOR	= 2		    # Channel 2 is used for Y
X_SWITCH = 4		# X switches are wired for horizontal
Y_SWITCH = 3		# Y switches are wired for vertical
X_OFFSET = 16		# offset for motor 1 (X axis). It is 32 in the manual but it should be 16.
Y_OFFSET = 32		# offset for motor 2 (Y axis). It is 64 in the manual but it should be 32.
# Catcher Settings
CATCHER_W = 22.0    # catcher width in inches
CATCHER_H = 15.0    # catcher height in inches
X_INCH_COUNTS = 70		# # of encoder counts per linear inch = 4 (quadrature) * encoder pulses per linear inch.  This depends on the gear ratio
Y_INCH_COUNTS = 193	    # # of encoder counts per linear inch = 4 (quadrature) * encoder pulses per linear inch.  This depends on the gear ratio
X_MOTOR_PPR = 88	    # Motor pulses per revolution.  Controller get 4 counts per pulse.
Y_MOTOR_PPR = 180        # Motor pulses per revolution.  Controller get 4 counts per pulse.
X_HOME_SPEED = -250     # Open loop speed limit to home
Y_HOME_SPEED = -250     # Open loop speed limit to home

# Close Loop Parameters
X_POS_VEL = 4000    # Position Mode Velocity (RPM)
X_TURN_MM = 1500    # Position Turns Min to Max
X_PID_KP = 200		# 200 Proportional Gain, 0~250 and 150 means 15.0
X_PID_KI = 0		# Integral Gain, 0~250 and 150 means 15.0
X_PID_KD = 0		# Differential Gain, 0~250 and 150 means 15.0
X_PID_ICAP = 100	# Integrator Limit (%)
# Close Loop Parameters
Y_POS_VEL = 4000    # Position Mode Velocity (RPM)
Y_TURN_MM = 1000    # Position Turns Min to Max
Y_PID_KP = 200		# 150 Proportional Gain, 0~250 and 150 means 15.0
Y_PID_KI = 0		# Integral Gain, 0~250 and 150 means 15.0
Y_PID_KD = 0		# 200 Differential Gain, 0~250 and 150 means 15.0
Y_PID_ICAP = 100	# Integrator Limit (%)

# General Settings Same for both Motors
PWM_FREQ = 180      # PWM Frequency
OVER_VOLTAGE = 350  # Over Voltage Limit
UNDER_VOLTAGE = 50  # Over Voltage Limit
AMPS_LIMIT = 200    # Amps Limit
INIT_POWER = 100	# Initial power output %
MAX_POWER = 100		# Maximum output power %
MAX_RPM	 = 4000	        # Maximum motor RPM at full power that is to be used as +1,000 in relative
ACCELERATION = 500000	# Must be 100~32,000.  Countedd as 0.1RPM per second, e.g., 10,000 means it will acelerate 10,000x0.1=1,000 RPM per second.
DECELERATION = 500000	# Must be 100~32,000.  Countedd as 0.1RPM per second, e.g., 10,000 means it will decelerate 10,000x0.1=1,000 RPM per second.
