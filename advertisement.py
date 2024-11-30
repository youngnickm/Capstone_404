import dbus
import dbus.exceptions
import dbus.mainloop.glib
import dbus.service
import os
import re
import sys
import subprocess
import time
import threading
import logging
import logging.handlers
#import board
#from adafruit_motorkit import MotorKit
#import RPi.GPIO as GPIO
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

import firebase_admin
from firebase_admin import credentials, storage

# Path to your service account key file
cred = credentials.Certificate("/home/team1/firebase_audio/firebase-key.json")

# Initialize the Firebase Admin SDK
firebase_admin.initialize_app(cred, {
    "storageBucket": "livetestv3.firebasestorage.app"
})

# Get a reference to the storage bucket
bucket = storage.bucket()

# Define the folder path to filter by
folder_path = "users/llalQtgJG3e3hSu0PST7tW0QoEG3/uploads/"

# Path to store the last downloaded file's timestamp
last_timestamp_path = "/home/team1/ble-uart-peripheral/audio_files/last_timestamp.txt"

sys.path.append(os.path.expanduser('~/Documents'))
import cont_funcs
from cont_funcs import read_journalctl

from gi.repository import GLib

# BlueZ D-Bus interfaces
BLUEZ_SERVICE_NAME = "org.bluez"
GATT_MANAGER_IFACE = "org.bluez.GattManager1"
DBUS_OM_IFACE = "org.freedesktop.DBus.ObjectManager"
DBUS_PROP_IFACE = "org.freedesktop.DBus.Properties"

GATT_SERVICE_IFACE = "org.bluez.GattService1"
GATT_CHRC_IFACE = "org.bluez.GattCharacteristic1"
GATT_DESC_IFACE = "org.bluez.GattDescriptor1"
LE_ADVERTISING_MANAGER_IFACE = 'org.bluez.LEAdvertisingManager1'
LE_ADVERTISEMENT_IFACE = 'org.bluez.LEAdvertisement1'

# Test UUIDs
TEST_SERVICE = "0000ffff-beef-c0c0-c0de-c0ffeefacade"
TEST_CHARACTERISTIC = "0000bbbb-beef-c0c0-c0de-c0ffeefacade"


# Boiler plate start
class InvalidArgsException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.freedesktop.DBus.Error.InvalidArgs'


class NotSupportedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.NotSupported'


class NotPermittedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.NotPermitted'


class InvalidValueLengthException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.InvalidValueLength'


class FailedException(dbus.exceptions.DBusException):
    _dbus_error_name = 'org.bluez.Error.Failed'


def register_app_cb():
    print("GATT application registered")


def register_app_error_cb(error):
    print("Failed to register application: " + str(error))
    mainloop.quit()


def register_ad_cb():
    print('Advertisement registered')


def register_ad_error_cb(error):
    print('Failed to register advertisement: ' + str(error))
    mainloop.quit()


class Advertisement(dbus.service.Object):
    PATH_BASE = '/org/bluez/example/advertisement'

    def __init__(self, bus, index, advertising_type):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.ad_type = advertising_type
        self.service_uuids = None
        self.manufacturer_data = None
        self.solicit_uuids = None
        self.service_data = None
        self.local_name = None
        self.include_tx_power = False
        self.data = None
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        properties = dict()
        properties['Type'] = self.ad_type
        if self.service_uuids is not None:
            properties['ServiceUUIDs'] = dbus.Array(self.service_uuids,
                                                    signature='s')
        if self.solicit_uuids is not None:
            properties['SolicitUUIDs'] = dbus.Array(self.solicit_uuids,
                                                    signature='s')
        if self.manufacturer_data is not None:
            properties['ManufacturerData'] = dbus.Dictionary(
                self.manufacturer_data, signature='qv')
        if self.service_data is not None:
            properties['ServiceData'] = dbus.Dictionary(self.service_data,
                                                        signature='sv')
        if self.local_name is not None:
            properties['LocalName'] = dbus.String(self.local_name)
        if self.include_tx_power:
            properties['Includes'] = dbus.Array(["tx-power"], signature='s')

        if self.data is not None:
            properties['Data'] = dbus.Dictionary(
                self.data, signature='yv')
        return {LE_ADVERTISEMENT_IFACE: properties}

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service_uuid(self, uuid):
        if not self.service_uuids:
            self.service_uuids = []
        self.service_uuids.append(uuid)

    def add_solicit_uuid(self, uuid):
        if not self.solicit_uuids:
            self.solicit_uuids = []
        self.solicit_uuids.append(uuid)

    def add_manufacturer_data(self, manuf_code, data):
        if not self.manufacturer_data:
            self.manufacturer_data = dbus.Dictionary({}, signature='qv')
        self.manufacturer_data[manuf_code] = dbus.Array(data, signature='y')

    def add_service_data(self, uuid, data):
        if not self.service_data:
            self.service_data = dbus.Dictionary({}, signature='sv')
        self.service_data[uuid] = dbus.Array(data, signature='y')

    def add_local_name(self, name):
        if not self.local_name:
            self.local_name = ""
        self.local_name = dbus.String(name)

    def add_data(self, ad_type, data):
        if not self.data:
            self.data = dbus.Dictionary({}, signature='yv')
        self.data[ad_type] = dbus.Array(data, signature='y')

    @dbus.service.method(DBUS_PROP_IFACE,
                         in_signature='s',
                         out_signature='a{sv}')
    def GetAll(self, interface):
        print('GetAll')
        if interface != LE_ADVERTISEMENT_IFACE:
            raise InvalidArgsException()
        print('returning props')
        return self.get_properties()[LE_ADVERTISEMENT_IFACE]

    @dbus.service.method(LE_ADVERTISEMENT_IFACE,
                         in_signature='',
                         out_signature='')
    def Release(self):
        print('%s: Released!' % self.path)


class Service(dbus.service.Object):
    """
    org.bluez.GattService1 interface implementation
    """

    PATH_BASE = "/org/bluez/app/service"

    def __init__(self, bus, index, uuid, primary):
        self.path = self.PATH_BASE + str(index)
        self.bus = bus
        self.uuid = uuid
        self.primary = primary
        self.characteristics = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_SERVICE_IFACE: {
                "UUID": self.uuid,
                "Primary": self.primary,
                "Characteristics": dbus.Array(
                    self.get_characteristic_paths(), signature="o"
                ),
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_characteristic(self, characteristic):
        self.characteristics.append(characteristic)

    def get_characteristic_paths(self):
        result = []
        for chrc in self.characteristics:
            result.append(chrc.get_path())
        return result

    def get_characteristics(self):
        return self.characteristics

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != GATT_SERVICE_IFACE:
            raise InvalidArgsException()

        return self.get_properties()[GATT_SERVICE_IFACE]


class Characteristic(dbus.service.Object):
    """
    org.bluez.GattCharacteristic1 interface implementation
    """

    def __init__(self, bus, index, uuid, flags, service):
        self.path = service.path + "/chrc" + str(index)
        self.bus = bus
        self.uuid = uuid
        self.service = service
        self.flags = flags
        self.descriptors = []
        dbus.service.Object.__init__(self, bus, self.path)

    def get_properties(self):
        return {
            GATT_CHRC_IFACE: {
                "Service": self.service.get_path(),
                "UUID": self.uuid,
                "Flags": self.flags,
                "Descriptors": dbus.Array(self.get_descriptor_paths(), signature="o"),
            }
        }

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_descriptor(self, descriptor):
        self.descriptors.append(descriptor)

    def get_descriptor_paths(self):
        result = []
        for desc in self.descriptors:
            result.append(desc.get_path())
        return result

    def get_descriptors(self):
        return self.descriptors

    @dbus.service.method(DBUS_PROP_IFACE, in_signature="s", out_signature="a{sv}")
    def GetAll(self, interface):
        if interface != GATT_CHRC_IFACE:
            raise InvalidArgsException()

        return self.get_properties()[GATT_CHRC_IFACE]

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="a{sv}", out_signature="ay")
    def ReadValue(self, options):
        print("Default ReadValue called, returning error")
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE, in_signature="aya{sv}")
    def WriteValue(self, value, options):
        print("Default WriteValue called, returning error")
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StartNotify(self):
        print("Default StartNotify called, returning error")
        raise NotSupportedException()

    @dbus.service.method(GATT_CHRC_IFACE)
    def StopNotify(self):
        print("Default StopNotify called, returning error")
        raise NotSupportedException()

    @dbus.service.signal(DBUS_PROP_IFACE, signature="sa{sv}as")
    def PropertiesChanged(self, interface, changed, invalidated):
        pass


def find_adapter(bus, iface):
    remote_om = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, '/'),
                               DBUS_OM_IFACE)
    objects = remote_om.GetManagedObjects()

    for o, props in objects.items():
        if iface in props:
            return o

    return None

# Boiler plate end


class TestService(Service):
    """
    Test service that provides a characteristic
    """

    def __init__(self, bus, index):
        Service.__init__(self, bus, index, TEST_SERVICE, True)
        self.add_characteristic(TestCharacteristic(bus, 0, self))


class TestCharacteristic(Characteristic):
    """
    Test characteristic. Allows writing arbitrary bytes to its value
    """

    def __init__(self, bus, index, service):
        Characteristic.__init__(
            self, bus, index, TEST_CHARACTERISTIC, ["write"], service
        )
        self.value = ""

    def WriteValue(self, value, options):
        #print(f"TestCharacteristic Write: {value}")
        txt = bytes(value).decode('utf8')
        print(f"As text: {txt}")
        self.value = txt
        my_write_callback(txt)


class TestAdvertisement(Advertisement):

    def __init__(self, bus, index):
        Advertisement.__init__(self, bus, index, 'peripheral')
        self.add_local_name('RoboTAMU')
        self.include_tx_power = True


class Application(dbus.service.Object):
    """
    org.bluez.GattApplication1 interface implementation
    """

    def __init__(self, bus):
        self.path = "/"
        self.services = []
        dbus.service.Object.__init__(self, bus, self.path)
        self.add_service(TestService(bus, 0))

    def get_path(self):
        return dbus.ObjectPath(self.path)

    def add_service(self, service):
        self.services.append(service)

    @dbus.service.method(DBUS_OM_IFACE, out_signature="a{oa{sa{sv}}}")
    def GetManagedObjects(self):
        response = {}
        print("GetManagedObjects")

        for service in self.services:
            response[service.get_path()] = service.get_properties()
            chrcs = service.get_characteristics()
            for chrc in chrcs:
                response[chrc.get_path()] = chrc.get_properties()
                descs = chrc.get_descriptors()
                for desc in descs:
                    response[desc.get_path()] = desc.get_properties()

        return response

def my_write_callback(txt):
    print(f"This is where I can use the <<{txt}>> value")
    # Execute the appropriate command based on the number received#
    if txt == '0':
        stop()
    elif txt == '1':
        forward(6.0, "inches")
    elif txt == '2':
        backward(6.0, "inches")
    elif txt == '3':
        left(45.0)
    elif txt == '4':
        right(45.0)
    elif txt == '5':
        while (True):
            download_mp3_file()
            time.sleep(3)
            if txt == '0':
                break

SPEED_DIST = 0.4 # Constant for speed of motors for lateral movement
SPEED_ROTA = 230.0 # Constant for speed of motors for rotational movement
SENSOR_DIST_THRESH = 10 # Constant for distance for sensor to stop robot from object

# Sensor Pins
'''
front_trig_pin = 27
front_echo_pin = 22
back_trig_pin = 23
back_echo_pin = 24

GPIO.setmode(GPIO.BCM)
GPIO.setup(front_trig_pin, GPIO.OUT)
GPIO.setup(front_echo_pin, GPIO.IN)
GPIO.setup(back_trig_pin, GPIO.OUT)
GPIO.setup(back_trig_pin, GPIO.IN)
'''
venv_path = '/home/team1/stt_env/bin/activate'
script_to_run = '/home/team1/STT_code/TranscriptionTesterV2.py'

command = f"source {venv_path} && python {script_to_run}"

force_stop = False # Emergency stop if sensor detects obstacle
sensor_active = False # Turns on sensor
sensor_stop = False # Turns off sensor
in_return = False # Checks if currently running the return movement function
return_fail = False # Variable to tell if the return movement function fails due to imminent collision
in_forward = True # Checks direction of movement for sensors

heading = 0.0 # Direction the robot is facing in degrees (0 to 359)
current_location = (5, 0) # Default starting position
in_goto = False # If currently using pathfinding
scale = 18

maze = [[        0, "Table_one", 0,           0, 0,             0],
        [        0,           1, 1,           0, 1,             0],
        [        0,           0, 0,           0, 1, "Table_three"],
        [        0,           1, 1, "Table_two", 1,             0],
        [        0,           1, 1,           0, 1,  "Table_four"],
        ["Kitchen",           0, 0,           0, 0,             0]]

curr_maze = []

#kit = MotorKit(i2c = board.I2C()) # Initalizes variable kit for all motors

conv_rate = { # Contains the conversions for all the distance units
    'meter': 1,
    'meters': 1,
    'inch': 0.0254,
    'inches': 0.0254,
    'centimeter': 0.01,
    'centimeters': 0.01,
    'foot': 0.3048,
    'feet': 0.3048,
    'yard': 0.9144,
    'yards': 0.9144
}

kwFor = ["forward", "forwards", "ahead"]
kwBac = ["backward", "backwards", "reverse"]
kwLef = ["left", "counter-clockwise"]
kwRig = ["right", "clockwise"]
kwSto = ["stop", "halt", "freeze"]
kwRet = ["return", "retrace"]
kwEnd = ["end", "finish"]
kwGoTo = ["goto", "to"]

keywords = [ # Contains the keywords for different movements
    "forward", 
    "forwards",
    "ahead",
    "backward",
    "backwards",
    "reverse",
    "left",
    "counter-clockwise",
    "right",
    "clockwise",
    "stop",
    "halt",
    "freeze",
    "return",
    "retrace",
    "end",
    "finish",
    "goto",
    "to"
]

units = [ # Contains the accepted unit names
    "meter",
    "meters",
    "inch",
    "inches",
    "centimeter",
    "centimeters",
    "foot",
    "feet",
    "yard",
    "yards"
]

spelled_numbers = { # Contains the word-form of numbers that can be combined up to 999
    'point': 0,
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90,
    'hundred': 100
}

class Node(): # Node Class for A* Pathing
    def __init__(self, parent = None, position = None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def prox_sens(): # Function for the proximity sensor (Needs updates when actual sensor is connected)
    global SENSOR_DIST_THRESH, force_stop, sensor_active, sensor_stop, current_location, in_goto, curr_maze
    while not force_stop:
        if sensor_active:
            distance = 15 # Simulated object distance
            '''
            if (in_forward):
                distance = measure_distance(front_trig_pin, front_echo_pin)
            else:
                distance = measure_distance(back_trig_pin, back_echo_pin)
            '''
            if distance < SENSOR_DIST_THRESH:
                print("Object Detected in Path!")
                sensor_stop = True
                stop();
                if in_goto:
                    if heading == 0.0:
                        curr_maze[current_location[0] - 1][current_location[1]] = 1
                    if heading == 90.0:
                        curr_maze[current_location[0][current_location[1] + 1]] = 1
                    if heading == 180.0:
                        curr_maze[current_location[0] + 1][current_location[1]] = 1
                    if heading == 270.0:
                        curr_maze[current_location[0][current_location[1] - 1]] = 1
                    print(f"Marking position {position} as non-traversable.")
                    pathing(destination)
            else:
                sensor_stop = False # Reset if no obstacle detected
        time.sleep(0.25)

def measure_distance(trig_pin, echo_pin):
    GPIO.output(trig_pin, True)
    time.sleep(0.00001)
    GPIO.output(trig_pin, False)

    while GPIO.input(echo_pin) == 0:
        start_time = time.time()
    while GPIO.input(echo_pin) == 1:
        end_time = time.time()
        
    elapsed_time = end_time - start_time
    distance = (elapsed_time * 34300) / 2
    return distance

def calc_dur(distance, unit): # Calculates duration of motor usaged based on desired distance to travel
    global conv_rate
    meters = distance * conv_rate[unit]
    duration = meters / SPEED_DIST
    return duration

def dur_to_dis(duration, unit): # Calculates actual distance traveled based on duration of motor activation
    global conv_rate
    meters = duration * SPEED_DIST
    distance = meters / conv_rate[unit]
    return distance

def calc_rota_dur(angle): # Calculates duration of motor activation based on number of degrees needing to rotate
    duration = angle / SPEED_ROTA
    return duration

def comp_num_to_int(text): # Converts Compound Text into Numeric Values
    tokens = text.split('-')
    total = 0
    for token in tokens:
        if token.lower() in spelled_numbers:
            if token.lower() == 'hundred':
                total = total * 100
            else:
                total += spelled_numbers[token.lower()]
        else:
            return None
    return total

def translate(text): # Takes multiple formats for command inputs to create proper command functions
    regex_patterns1 = [re.compile(r'\b{}\b'.format(re.escape(word)), re.IGNORECASE) for word in keywords] # Formats for searching for keywords
    regex_patterns2 = [re.compile(r'\b{}\b'.format(re.escape(word)), re.IGNORECASE) for word in units] # Formats for searching for units

    input_words = re.findall(r'\b[\w-]+\b', text.lower()) # Seperates and normalizes words in input text

    found_keyword = None
    found_unit = None
    is_decimal = False
    num_keywords = 0
    num_units = 0
    found_go_to = False

    for word in input_words: # Loops through all words
        for pattern in regex_patterns1: # Searches through all keywords
            if pattern.search(word): # If keyword found
                num_keywords += 1
                if num_keywords >= 2: # If more than 1 keyword, do not accept command
                    return "Too Many Commands"
                found_keyword = word
        for pattern in regex_patterns2: # Searches through all units
            if pattern.search(word): # If unit found
                num_units += 1
                if num_units >= 2: # If more than 1 unit, do not accept command
                    return "Too Many Units"
                found_unit = word
        # Check for "to"
        if 'to' in input_words:
            found_go_to = True

    # Handle go to command
    if found_go_to:
        location = ' '.join(input_words[input_words.index('to') + 1:]) # Get the location after "go to"
        return f"to {location}"
    
    if found_keyword is None: # If no keyword, do not accept command
        return "No Keyword Found"

    numbers = [] # Creates list for Numeric Values
    #spelled_out = set(re.findall(r'\b[a-zA-Z]+\b', text)) # Creates List of Spelled-Out Numbers

    for word in input_words:
        if '-' in word:
            compound_num = comp_num_to_int(word)
            if compound_num is not None:
                numbers.append(str(compound_num))
        if word.lower() == 'point':
            numbers.append(str('point'))
        elif word.lower() in spelled_numbers:
            numbers.append(str(spelled_numbers[word.lower()]))

    if numbers: # Combines multiple words in place value notation
        total = 0
        is_deci = 1.0
        for num in numbers:
            if num.isdigit():
                if int(num) > 9 and int(num) < 100:
                    is_deci = is_deci #* 0.1
                if num == '100':
                    total = total * 100 * is_deci
                else:
                    total += int(num) * is_deci
            if num == 'point' or is_deci < 1:
                is_deci = is_deci * 0.1
        numbers.clear()
        numbers.append(total)

    if any(found_keyword in string for string in kwEnd):
        return "end"
    elif any(found_keyword in string for string in kwRet):
        return "return"
    elif any(found_keyword in string for string in kwSto):
        return "stop"

    if not numbers:
        return "No Value Found"

    if any(found_keyword in string for string in kwFor):
        if found_unit is None:
            return f"No Recognized Unit Found For '{found_keyword}' Command."
        return f"forward {numbers.pop()} {found_unit}"
    elif any(found_keyword in string for string in kwBac):
        if found_unit is None:
            return f"No Recognized Unit Found For '{found_keyword}' Command."
        return f"backward {numbers.pop()} {found_unit}"
    elif any(found_keyword in string for string in kwLef):
        return f"left {numbers.pop()}"
    elif any(found_keyword in string for string in kwRig):
        return f"right {numbers.pop()}"

def write_to_return_temp(command, distance=0, unit=None): # Keeps track of commands for return function
    if not in_return: # Does not record movements during return function
        with open("returnTemp.txt", 'a') as returnFile:
            if command == 'forward':
                returnFile.write(f"backward {distance} {unit}\n")
            elif command == 'backward':
                returnFile.write(f"forward {distance} {unit}\n")
            elif command == 'left':
                returnFile.write(f"right {distance}\n")
            elif command == 'right':
                returnFile.write(f"left {distance}\n")
            returnFile.close()

def stop(): # Command for stopping the robot
    print("Stopping.")
    
    #kit.motor1.throttle = 0
    #kit.motor2.throttle = 0
    
    time.sleep(0.5)
    force_stop = False

def forward(distance, unit): # Command for moving the robot forward
    global force_stop, sensor_active, in_goto, scale, heading, current_location, in_forward
    in_forward = True
    sensor_active = True
    if in_goto:
        steps = distance/scale
        stepsDur = calc_dur(scale, unit)
        prevStep = 0
    duration = calc_dur(distance, unit)
    print(f"Moving Forward {distance} {unit}.")
    
    #kit.motor1.throttle = 1
    #kit.motor2.throttle = -1
    
    start_time = time.time()
    elapsed = 0

    while time.time() - start_time < duration and not force_stop: # Checks for possible collisions
        elapsed = time.time() - start_time # If force stopped, records only elapsed time
        if in_goto:
            if prevStep + 1 <= elapsed / stepsDur:
                prevStep += 1
                if heading == 0.0:
                    current_location = (current_location[0] - 1, current_location[1])
                elif heading == 90.0:
                    current_location = (current_location[0], current_location[1] + 1)
                elif heading == 180.0:
                    current_location = (current_location[0] + 1, current_location[1])
                elif heading == 270.0:
                    current_location = (current_location[0], current_location[1] - 1)
                print(current_location)
        if sensor_stop: # If an obstacle is detected
            print("Obstacle detected, pausing for 2 seconds...")
            stop()
            time.sleep(2) # Wait 2 seconds
            start_time += 2

            # Check again for obstacle after waiting
            if sensor_stop: # If obstacle is still there
                print("Obstacle still present, stopping movement.")
                if in_goto:
                    print("Rerouting due to obstacle...")
                    elapsed = time.time() - start_time
                    stepProg = elapsed / stepsDur - prevStep
                    backward(dur_to_dis(stepProg, "inches"), "inches")
                    pathing(destination)
                break # Stop the movement
            else:
                print("Obstacle cleared, resuming movement.")
                
                #kit.motor1.throttle = 1
                #kit.motor2.throttle = -1
                    

        time.sleep(0.1)
    if in_goto:
        if heading == 0.0:
            current_location = (current_location[0] - 1, current_location[1])
        elif heading == 90.0:
            current_location = (current_location[0], current_location[1] + 1)
        elif heading == 180.0:
            current_location = (current_location[0] + 1, current_location[1])
        elif heading == 270.0:
            current_location = (current_location[0], current_location[1] - 1)
        print(current_location)
    if not force_stop: # If no forced stop, take full duration
        elapsed = duration
    if force_stop and in_return:
        return_fail = True
    sensor_active = False
    stop()
    write_to_return_temp('forward', dur_to_dis(elapsed, unit), unit)

def backward(distance, unit): # Command for moving the robot backward
    global force_stop, sensor_active, in_forward
    in_forward = False
    sensor_active = True
    duration = calc_dur(distance, unit)
    print(f"Moving Backward {distance} {unit}.")
    
    #kit.motor1.throttle = -1
    #kit.motor2.throttle = 1
    
    start_time = time.time()
    elapsed = 0

    while time.time() - start_time < duration and not force_stop: # Checks for possible collisions
        elapsed = time.time() - start_time # If force stopped, records only elapsed time
        if sensor_stop: # If an obstacle is detected
            print("Obstacle detected, pausing for 2 seconds...")
            stop()
            time.sleep(2) # Wait for 2 seconds
            start_time += 2

            if sensor_stop: # If obstacle is still there
                print("Obstacle still present, stopping movement.")
                if in_goto:
                    execute_return_commands() # Return to previous marker
                    pathing(destination)
                break # Stop the movement
            else:
                print("Obstacle cleared, resuing movement.")
                
                #kit.motor1.throttle = -1
                #kit.motor2.throttle = 1

        time.sleep(0.1)
    if not force_stop: # If no forced stop, take full duration
        elapsed = duration
    if force_stop and in_return:
        return_fail = True
    sensor_active = False
    stop()
    write_to_return_temp('backward', dur_to_dis(elapsed, unit), unit)

def left(angle): # Command for turn the robot left
    global heading
    duration = calc_rota_dur(angle)
    print(f"Rotating Left {angle} Degrees.")
    
    #kit.motor1.throttle = 1
    #kit.motor2.throttle = 1
    
    heading -= angle
    while heading < 0:
        heading += 360

    time.sleep(duration)
    stop()
    write_to_return_temp('left', angle)

def right(angle): # Command for turn the robot right
    global heading
    duration = calc_rota_dur(angle)
    print(f"Rotating Right {angle} Degrees.")
    
    #kit.motor1.throttle = -1
    #kit.motor2.throttle = -1
    
    heading += angle
    while heading >= 360:
        heading -= 360

    time.sleep(duration)
    stop()
    write_to_return_temp('right', angle)

def execute_return_commands(): # Processes return function
    global in_return
    print("Executing return commands.")
    in_return = True
    with open("returnTemp.txt", 'r') as returnFile:
        commands = returnFile.readlines()
        for command in reversed(commands):
            if return_fail:
                print("Object Detected in Return Path.\n")
                print("Cancelling Return Function.\n")
                break
            if command == '\n':
                break

            command = command.strip().split()
            
            if command[0] == 'forward':
                forward(float(command[1]), command[2])
            elif command[0] == 'backward':
                backward(float(command[1]), command[2])
            elif command[0] == 'left':
                left(float(command[1]))
            elif command[0] == 'right':
                right(float(command[1]))
        returnFile.close()
    open("returnTemp.txt", 'w').close()
    in_return = False

def control(file_path=None):
    global force_stop

    open("returnTemp.txt", 'w').close() # Clears returnTemp in case of lingering commands from previous runs

    if file_path:
        run_commands_from_file(file_path)
    else:
        while not force_stop:
            command = translate(input("Enter a command (or 'end' to exit): ")).strip().split()
            if command[0] == 'No' or command[0] == 'Too':
                err = ' '.join(command)
                print(f"Error: {err}")
            elif command[0] == 'end':
                force_stop = True
                break
            elif command[0] in ['forward', 'backward', 'left', 'right', 'return', 'to']:
                execute_command(command)
            else:
                print(f"Unknown command: {command[0]}.")
    print("Exiting Program.")

def run_commands_from_file(file_path):
    global force_stop
    with open(file_path, 'r') as file:
        while not force_stop:
            for line in file:
                if force_stop:
                    break
                command = translate(line).split()
                if command[0] == 'end':
                    force_stop = True
                    break
                execute_command(command)
            file.seek(0)
    file.close()
    print("End Command Received. Exiting Program.")

def execute_command(command):
    global force_stop, current_location
    if command[0] == 'to':
        in_goto = True
        location = ' '.join(command[1:]) # Get the full location string
        execute_return_commands()
        pathing(location) # Call pathing with the specified location
        in_goto = False
    elif command[0] == 'forward':
        forward(float(command[1]), command[2])
    elif command[0] == 'backward':
        backward(float(command[1]), command[2])
    elif command[0] == 'left':
        left(float(command[1]))
    elif command[0] == 'right':
        right(float(command[1]))
    elif command[0] == 'stop':
        stop()
    elif command[0] == 'return':
        #execute_return_commands()
        in_goto = True
        execute_return_commands()
        pathing("Kitchen") # Call pathing with the specified location
        in_goto = False
    elif command[0] == 'end':
        force_stop = True
        return
    else:
        print(f"Unknown command: {command[0]}.")

def astar(maze, start, end): # Returns list as a path from start to end in given maze
    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while open_list:
        # Get the current node
        current_node = min(open_list, key=lambda node: node.f)
        
        open_list.remove(current_node)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]: # Adjacent squares
            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Ensure within range and walkable terrain
            if (0 <= node_position[0] < len(maze) and 0 <= node_position[1] < len(maze[0]) and (maze[node_position[0]][node_position[1]] == 0 or isinstance(maze[node_position[0]][node_position[1]], str))):
                # Create new node
                new_node = Node(current_node, node_position)

                # Skip if already in closed list
                if new_node in closed_list:
                    continue

                # Create the f, g, and h values
                new_node.g = current_node.g + 1
                new_node.h = ((new_node.position[0] - end_node.position[0]) ** 2) + ((new_node.position[1] - end_node.position[1]) ** 2)
                new_node.f = new_node.g + new_node.h

                # Add the child to the open list
                if new_node not in open_list:
                    open_list.append(new_node)

    return None

def find_position(maze, word):
    for i, row in enumerate(maze):
        for j, value in enumerate(row):
            if str(value).lower() == word.lower():
                return (i, j)
    return None

def pathing(destination):
    global current_location, in_goto, maze, curr_maze
    in_goto = True

    curr_maze = maze

    start = current_location
    end_word = destination.strip().replace(" ", "_") # Change format for consistency

    #start = find_position(maze, start_word)
    end = find_position(curr_maze, end_word)

    if start is not None and end is not None:
        #print(start)
        #print(end)
        path = astar(curr_maze, start, end)
        if path:
            directions(path)
            current_location = end # Update current location
            print("Arrived at " + end_word)

            open("returnTemp.txt", 'w').close() # Clear the return file when reaching the desitination
        else:
            print("No path found.")
    else:
        print("Start or end word not found in the maze.")

def directions(path):
    global SENSOR_DIST_THRESH, sensor_stop, scale
    counter = 0
    start = path[0]
    cumulative_dist = 0 # Distance for consecutive forward commands
    for node in path[1:]:
        prevHeading = heading
        if start[0] < node[0] and start[1] == node[1]:
            # print("Down")
            needed_heading = 180.0
        elif start[0] == node[0] and start[1] < node[1]:
            # print("Right")
            needed_heading = 90.0
        elif start[0] == node[0] and start[1] > node[1]:
            # print("Left")
            needed_heading = 270.0
        elif start[0] > node[0] and start[1] == node[1]:
            # print("Up")
            needed_heading = 0.0

        if heading != needed_heading: # If about to change heading
            if cumulative_dist > 0: # If distance needed to travel
                forward(cumulative_dist, "inches")
                cumulative_dist = 0 # Reset distance
        
        if (heading - needed_heading) > 0:
            left(heading - needed_heading)
        elif (heading + 360 - needed_heading) <= 180:
            left(heading + 360 - needed_heading)
        elif(needed_heading - heading) > 0:
            right(needed_heading - heading)
        
        cumulative_dist += scale

        if prevHeading != heading:
            if (heading == 0):
                print("Heading: North")
            elif (heading == 90):
                print("Heading: East")
            elif (heading == 180):
                print("Heading: South")
            elif (heading == 270):
                print("Heading: West")

        start = node
    if cumulative_dist > 0:
        forward(cumulative_dist, "inches")

def get_last_timestamp():
    # Read the last timestamp from file
    if os.path.exists(last_timestamp_path):
        with open(last_timestamp_path, "r") as f:
            return f.read().strip()
    return None

def set_last_timestamp(timestamp):
    # Write the latest timestamp to file
    with open(last_timestamp_path, "w") as f:
        f.write(timestamp)

def download_mp3_file():
    # List all files in the bucket and filter by folder path
    blobs = bucket.list_blobs()
    filtered_files = [blob for blob in blobs if blob.name.startswith(folder_path)]

    # Sort files by update time in descending order (most recent first)
    sorted_files = sorted(filtered_files, key=lambda b: b.updated, reverse=True)

    # Get the last timestamp from file
    last_timestamp = get_last_timestamp()
    
    if sorted_files:
        most_recent_file = sorted_files[0]
        most_recent_timestamp = most_recent_file.updated.isoformat()

        # Check if the most recent file is newer than the last downloaded one
        if last_timestamp is None or most_recent_timestamp > last_timestamp:
            print(f"New file detected: {most_recent_file.name}")
            most_recent_file.download_to_filename("/home/team1/ble-uart-peripheral/audio_files/recent.mp3")

            # Update the last timestamp
            set_last_timestamp(most_recent_timestamp)
            print("File downloaded and timestamp updated.")
            convert_to_flac("/home/team1/ble-uart-peripheral/audio_files/recent.mp3")
        else:
            print("No new file detected.")
    else:
        print("No files found.")

def convert_to_flac(input_file):
    # Define the output file name
    base, _ = os.path.splitext(input_file)
    output_file = f"{base}_16k.mp3"

    try:
        command = [
            "ffmpeg", "-y", "-i", input_file, "-ar", "16000", "-filter:a", "volume=2", "-ac", "1", output_file
        ]
        subprocess.run(command, check=True)
        print(f"File converted successfully with FFmpeg: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting file with FFmpeg: {e}")
    
    command = "source /home/team1/stt_env/bin/activate && python /home/team1/STT_code/TranscriptionTesterV2.py && deactivate"
    
    # Execute the command
    try:
        print("Working...")
        gospel = subprocess.run(command, capture_output=True, text=True, shell=True)
        print(gospel.stdout)
        execute_command(translate(gospel))
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def main():
    global mainloop

    sensor_thread = threading.Thread(target=prox_sens, daemon = True)
    sensor_thread.start()

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

    bus = dbus.SystemBus()

    adapter = find_adapter(bus, LE_ADVERTISING_MANAGER_IFACE)
    if not adapter:
        print("Adapter not found")
        return

    service_manager = dbus.Interface(
        bus.get_object(BLUEZ_SERVICE_NAME, adapter), GATT_MANAGER_IFACE
    )

    app = Application(bus)

    test_advertisement = TestAdvertisement(bus, 0)

    ad_manager = dbus.Interface(bus.get_object(BLUEZ_SERVICE_NAME, adapter),
                                LE_ADVERTISING_MANAGER_IFACE)

    ad_manager.RegisterAdvertisement(test_advertisement.get_path(), {},
                                     reply_handler=register_ad_cb,
                                     error_handler=register_ad_error_cb)
    mainloop = GLib.MainLoop()

    print("Registering GATT application...")

    service_manager.RegisterApplication(
        app.get_path(),
        {},
        reply_handler=register_app_cb,
        error_handler=register_app_error_cb,
    )

    mainloop.run()



if __name__ == "__main__":
    main()
