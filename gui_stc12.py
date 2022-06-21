from copyreg import remove_extension
from faulthandler import disable
import glob
import os
import cv2
import threading
import torch
import numpy as np 
import time

import PySimpleGUI as sg

from PIL import Image,ImageTk
import os
import datetime 
import shutil

from PIL import Image

from udp import UDPFinsConnection
from initialization import FinsPLCMemoryAreas

import traceback

import stapipy as st

#CAM 1   2048x1536              192.168.25.2
#CAM 2   1440x1080              169.254.139.50


# SCALE_X_CAM1 = 1/3.2
# SCALE_Y_CAM1 = 1/3.2

# SCALE_X_CAM2 = 1/2.25
# SCALE_Y_CAM2 = 1/2.25


# SCALE_X_CAM1 = 640/2048
# SCALE_Y_CAM1 = 480/1536

mysleep = 0.1

SCALE_X_CAM1 = 640*1.2/2048
SCALE_Y_CAM1 = 480*1.2/1536

SCALE_X_CAM2 = 640/1440
SCALE_Y_CAM2 = 480/1080

class CMyCallback:
    """
    Class that contains a callback function.
    """

    def __init__(self):
        self._image = None
        self._lock = threading.Lock()

    @property
    def image(self):
        """Property: return PyIStImage of the grabbed image."""
        duplicate = None
        self._lock.acquire()
        if self._image is not None:
            duplicate = self._image.copy()
        self._lock.release()
        return duplicate

    def datastream_callback1(self, handle=None, context=None):
        """
        Callback to handle events from DataStream.

        :param handle: handle that `trigger` the callback.
        :param context: user data passed on during callback registration.
        """
        st_datastream = handle.module
        if st_datastream:
            with st_datastream.retrieve_buffer() as st_buffer:
                # Check if the acquired data contains image data.
                if st_buffer.info.is_image_present:
                    # Create an image object.
                    st_image = st_buffer.get_image()

                    # Check the pixelformat of the input image.
                    pixel_format = st_image.pixel_format
                    pixel_format_info = st.get_pixel_format_info(pixel_format)

                    # Only mono or bayer is processed.
                    if not(pixel_format_info.is_mono or pixel_format_info.is_bayer):
                        return

                    # Get raw image data.
                    data = st_image.get_image_data()

                    # Perform pixel value scaling if each pixel component is
                    # larger than 8bit. Example: 10bit Bayer/Mono, 12bit, etc.
                    if pixel_format_info.each_component_total_bit_count > 8:
                        nparr = np.frombuffer(data, np.uint16)
                        division = pow(2, pixel_format_info
                                       .each_component_valid_bit_count - 8)
                        nparr = (nparr / division).astype('uint8')
                    else:
                        nparr = np.frombuffer(data, np.uint8)

                    # Process image for display.
                    nparr = nparr.reshape(st_image.height, st_image.width, 1)

                    # Perform color conversion for Bayer.
                    if pixel_format_info.is_bayer:
                        bayer_type = pixel_format_info.get_pixel_color_filter()
                        if bayer_type == st.EStPixelColorFilter.BayerRG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_RG2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGR:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GR2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGB:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GB2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerBG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_BG2RGB)

                    # Resize image and store to self._image.
                    nparr = cv2.resize(nparr, None,
                                       fx=SCALE_X_CAM1,
                                       fy=SCALE_Y_CAM1)
                    self._lock.acquire()
                    self._image = nparr
                    self._lock.release()


    def datastream_callback2(self, handle=None, context=None):
        """
        Callback to handle events from DataStream.

        :param handle: handle that trigger the callback.
        :param context: user data passed on during callback registration.
        """
        st_datastream = handle.module
        if st_datastream:
            with st_datastream.retrieve_buffer() as st_buffer:
                # Check if the acquired data contains image data.
                if st_buffer.info.is_image_present:
                    # Create an image object.
                    st_image = st_buffer.get_image()

                    # Check the pixelformat of the input image.
                    pixel_format = st_image.pixel_format
                    pixel_format_info = st.get_pixel_format_info(pixel_format)

                    # Only mono or bayer is processed.
                    if not(pixel_format_info.is_mono or pixel_format_info.is_bayer):
                        return

                    # Get raw image data.
                    data = st_image.get_image_data()

                    # Perform pixel value scaling if each pixel component is
                    # larger than 8bit. Example: 10bit Bayer/Mono, 12bit, etc.
                    if pixel_format_info.each_component_total_bit_count > 8:
                        nparr = np.frombuffer(data, np.uint16)
                        division = pow(2, pixel_format_info
                                       .each_component_valid_bit_count - 8)
                        nparr = (nparr / division).astype('uint8')
                    else:
                        nparr = np.frombuffer(data, np.uint8)

                    # Process image for display.
                    nparr = nparr.reshape(st_image.height, st_image.width, 1)

                    # Perform color conversion for Bayer.
                    if pixel_format_info.is_bayer:
                        bayer_type = pixel_format_info.get_pixel_color_filter()
                        if bayer_type == st.EStPixelColorFilter.BayerRG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_RG2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGR:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GR2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerGB:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_GB2RGB)
                        elif bayer_type == st.EStPixelColorFilter.BayerBG:
                            nparr = cv2.cvtColor(nparr, cv2.COLOR_BAYER_BG2RGB)

                    # Resize image and store to self._image.
                    nparr = cv2.resize(nparr, None,
                                       fx=SCALE_X_CAM2,
                                       fy=SCALE_Y_CAM2)
                    self._lock.acquire()
                    self._image = nparr
                    self._lock.release()


def set_enumeration(nodemap, enum_name, entry_name):
    enum_node = st.PyIEnumeration(nodemap.get_node(enum_name))
    entry_node = st.PyIEnumEntry(enum_node[entry_name])
    enum_node.set_entry_value(entry_node)



def setup_camera1_stc():
    global error_cam1
    while error_cam1 == True:
        try:
            st_device1 = st_system.create_first_device()
            print('Device1=', st_device1.info.display_name)
            st_datastream1 = st_device1.create_datastream()
            callback1 = st_datastream1.register_callback(cb_func1)
            st_datastream1.start_acquisition()
            st_device1.acquisition_start()
            remote_nodemap1 = st_device1.remote_port.nodemap
            set_enumeration(remote_nodemap1,"TriggerMode", "Off")
            error_cam1 = False
            return  st_datastream1, st_device1,remote_nodemap1

        except Exception as exception:
            print(' Error Cam 1:', exception)
            str_error = "Error"
            window['result_cam1'].update(value= str_error, text_color='red',)



def setup_camera2_stc():
    global error_cam2
    while error_cam2 == True:
        try:
            st_device2 = st_system.create_first_device()
            print('Device2=', st_device2.info.display_name)
            st_datastream2 = st_device2.create_datastream()
            callback2 = st_datastream2.register_callback(cb_func2)
            st_datastream2.start_acquisition()
            st_device2.acquisition_start()
            remote_nodemap2 = st_device2.remote_port.nodemap
            set_enumeration(remote_nodemap2,"TriggerMode", "Off")
            error_cam2 = False
            return  st_datastream2, st_device2,remote_nodemap2
        except Exception as exception:     
            print('Error Cam 2:', exception)
            str_error = "Error"
            #sg.popup(str_error,font=('Helvetica',15), text_color='red',keep_on_top= True)
            window['result_cam2'].update(value= str_error, text_color='red')


def connect_plc(host):
    global fins_instance
    try:
        fins_instance = UDPFinsConnection()
        fins_instance.connect(host)
        fins_instance.dest_node_add=1
        fins_instance.srce_node_add=25

        return True
    except:
        print("Can't connect to PLC")
        for i in range(100000000):
            pass
        #sleep(3)
        print("Reconnecting....")
        return False

def removefile():

    directory3 = 'G:/FH/camera1/'
    directory4 = 'G:/FH/camera2/'
    if os.listdir(directory3) != []:
        for i in glob.glob(directory3+'*'):
            for j in glob.glob(i+'/*'):
                os.remove(j)
            os.rmdir(i)

    if os.listdir(directory4) != []:
        for i in glob.glob(directory4+'*'):
            for j in glob.glob(i+'/*'):
                os.remove(j)
            os.rmdir(i)
    print('already delete folder')


def time_to_name():
    current_time = datetime.datetime.now() 
    name_folder = str(current_time)
    name_folder = list(name_folder)
    for i in range(len(name_folder)):
        if name_folder[i] == ':':
            name_folder[i] = '-'
        if name_folder[i] == ' ':
            name_folder[i] ='_'
        if name_folder[i] == '.':
            name_folder[i] ='-'
    name_folder = ''.join(name_folder)
    return name_folder


def load_param_model(filename):
    list_param_model = []
    with open(filename) as lines:
        for line in lines:
            _, value = line.strip().split('=')
            list_param_model.append(value)
    return list_param_model


def save_param_model1_A22(weight,size,conf,nut_me,divat,me,namchamcao,traybactruc,divatduoi,kimcaomax,kimcaomin):
    line1 = 'weights1' + '=' + str(weight)
    line3 = 'size1' + '=' + str(size)
    line4 = 'conf1' + '=' + str(conf)
    line5 = 'nut_me1' + '=' + str(nut_me)
    line6 = 'divat1' + '=' + str(divat)
    line7 = 'me1' + '=' + str(me)
    line8 = 'namchamcao1' + '=' + str(namchamcao)
    line9 = 'traybactruc1' + '=' + str(traybactruc)
    line10 = 'divatduoi1' + '=' + str(divatduoi)
    line11 = 'kimcaomax1' + '=' + str(kimcaomax)
    line12 = 'kimcaomin1' + '=' + str(kimcaomin)

    lines = [line1,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12]
    with open('static/param_model1_A22.txt', "w") as f:
        for i in lines:
            f.write(i)
            f.write('\n')

def save_param_model1_A19(weight,size,conf,nut_me,divat,me,namchamcao,traybactruc,divatduoi,kimcaomax,kimcaomin):
    line2 = 'weights1' + '=' + str(weight)
    line3 = 'size1' + '=' + str(size)
    line4 = 'conf1' + '=' + str(conf)
    line5 = 'nut_me1' + '=' + str(nut_me)
    line6 = 'divat1' + '=' + str(divat)
    line7 = 'me1' + '=' + str(me)
    line8 = 'namchamcao1' + '=' + str(namchamcao)
    line9 = 'traybactruc1' + '=' + str(traybactruc)
    line10 = 'divatduoi1' + '=' + str(divatduoi)
    line11 = 'kimcaomax1' + '=' + str(kimcaomax)
    line12 = 'kimcaomin1' + '=' + str(kimcaomin)

    lines = [line2,line3,line4,line5,line6,line7,line8,line9,line10,line11,line12]
    with open('static/param_model1_A19.txt', "w") as f:
        for i in lines:
            f.write(i)
            f.write('\n')

def save_param_model2_A22(weight,size,conf,nut_me,divat,me):
    line1 = 'weights2' + '=' + str(weight)
    line3 = 'size2' + '=' + str(size)
    line4 = 'conf2' + '=' + str(conf)
    line5 = 'nut_me2' + '=' + str(nut_me)
    line6 = 'divat2' + '=' + str(divat)
    line7 = 'me2' + '=' + str(me)

    lines = [line1,line3,line4,line5,line6,line7]
    with open('static/param_model2_A22.txt', "w") as f:
        for i in lines:
            f.write(i)
            f.write('\n')

def save_param_model2_A19(weight,size,conf,nut_me,divat,me):
    line1 = 'weights2' + '=' + str(weight)
    line3 = 'size2' + '=' + str(size)
    line4 = 'conf2' + '=' + str(conf)
    line5 = 'nut_me2' + '=' + str(nut_me)
    line6 = 'divat2' + '=' + str(divat)
    line7 = 'me2' + '=' + str(me)

    lines = [line1,line3,line4,line5,line6,line7]
    with open('static/param_model2_A19.txt', "w") as f:
        for i in lines:
            f.write(i)
            f.write('\n')

def load_theme():
    name_themes = []
    with open('static/theme.txt') as lines:
        for line in lines:
            _, name_theme = line.strip().split(':')
            name_themes.append(name_theme)
    return name_themes

def save_theme(name_theme):
    line = 'theme:' + name_theme
    with open('static/theme.txt','w') as f:
        f.write(line)


def load_choose_model():
    name_models = []
    with open('static/choose_model.txt') as lines:
        for line in lines:
            _, name_model = line.strip().split(':')
            name_models.append(name_model)
    return name_models[0]

def save_choose_model(name_model):
    line = 'choose_model:' + name_model
    with open('static/choose_model.txt','w') as f:
        f.write(line)



def config_off_auto1(remote_nodemap):
    # Configure the ExposureMode
    node_name1 = "ExposureMode"

    node1 = remote_nodemap.get_node(node_name1)

    if not node1.is_writable:
        print('not node ExposureMode')
    enum_node1 = st.PyIEnumeration(node1)
    enum_entries1 = enum_node1.entries
    selection1 = 1

    if selection1 < len(enum_entries1):
        enum_entry1 = enum_entries1[selection1]
        enum_node1.set_int_value(enum_entry1.value)

    # Configure the ExposureAuto
    node_name2 = "ExposureAuto"

    node2 = remote_nodemap.get_node(node_name2)

    if not node2.is_writable:
        print('not node ExposureAuto')
    enum_node2 = st.PyIEnumeration(node2)
    enum_entries2 = enum_node2.entries
    selection2 = 0

    if selection2 < len(enum_entries2):
        enum_entry2 = enum_entries2[selection2]
        enum_node2.set_int_value(enum_entry2.value)

    #Configure the BalanceWhiteAuto
    node_name7 = "BalanceWhiteAuto"

    node7 = remote_nodemap.get_node(node_name7)

    if not node7.is_writable:
        print('not node 7')
    enum_node7 = st.PyIEnumeration(node7)
    enum_entries7 = enum_node7.entries
    selection7 = 0

    if selection7 < len(enum_entries7):
        enum_entry7 = enum_entries7[selection7]
        enum_node7.set_int_value(enum_entry7.value)
        

def config_off_auto2(remote_nodemap):
    # Configure the ExposureMode
    node_name1 = "ExposureMode"

    node1 = remote_nodemap.get_node(node_name1)

    if not node1.is_writable:
        print('not node ExposureMode')
    enum_node1 = st.PyIEnumeration(node1)
    enum_entries1 = enum_node1.entries
    selection1 = 1

    if selection1 < len(enum_entries1):
        enum_entry1 = enum_entries1[selection1]
        enum_node1.set_int_value(enum_entry1.value)

    # Configure the ExposureAuto
    node_name2 = "ExposureAuto"

    node2 = remote_nodemap.get_node(node_name2)

    if not node2.is_writable:
        print('not node ExposureAuto')
    enum_node2 = st.PyIEnumeration(node2)
    enum_entries2 = enum_node2.entries
    selection2 = 0

    if selection2 < len(enum_entries2):
        enum_entry2 = enum_entries2[selection2]
        enum_node2.set_int_value(enum_entry2.value)


def BalanceWhiteAuto(name, name1,name_button,index):
    if event == name1:
        values[name] = values[name1]
        window[name].update(value = values[name1])
    if event == name:
        values[name1] = values[name]
        window[name1].update(value = values[name])
        
    if event == name_button:
        if values[name] not in range(0,255):
            values[name] = values[name1]
            window[name].update(value = values[name1])
        enum_name1 = "BalanceRatioSelector"
        numeric_name1 = "BalanceRatio"
        node1 = remote_nodemap1.get_node(enum_name1)
        if not node1.is_writable:
            print('not node 1 '+ enum_name1)

        enum_node1 = st.PyIEnumeration(node1)
        enum_entries1 = enum_node1.entries

        enum_entry1 = enum_entries1[index]
        if enum_entry1.is_available:
            enum_node1.value = enum_entry1.value
            #print(st.PyIEnumEntry(enum_entry).symbolic_value)
            node_name2 = numeric_name1
            node2 = remote_nodemap1.get_node(node_name2)

            if not node2.is_writable:
                print('not node 2 '+ name)
            else:
                if node2.principal_interface_type == st.EGCInterfaceType.IFloat:
                    node_value2 = st.PyIFloat(node2)
                elif node2.principal_interface_type == st.EGCInterfaceType.IInteger:
                    node_value2 = st.PyIInteger(node2)
                value2 = values[name]

                if node2.principal_interface_type == st.EGCInterfaceType.IFloat:
                    value2 = float(value2)
                else:
                    value2 = int(value2)
                if node_value2.min <= value2 <= node_value2.max:
                    node_value2.value = value2


def set_exposure_or_gain(remote_nodemap,node_name,node_name1,value):
    if remote_nodemap.get_node(node_name):
        node = remote_nodemap.get_node(node_name)
        if not node.is_writable:
            print('not node' + node_name)
        else:
            if node.principal_interface_type == st.EGCInterfaceType.IFloat:
                node_value = st.PyIFloat(node)
            elif node.principal_interface_type == st.EGCInterfaceType.IInteger:
                node_value = st.PyIInteger(node)
            value = float(value)
            if node.principal_interface_type == st.EGCInterfaceType.IFloat:
                value = float(value)
            else:
                value = int(value)
            if node_value.min <= value <= node_value.max:
                node_value.value = value
            
            if remote_nodemap == remote_nodemap1:
                print(node_name + '1 : ' + str(node_value.value))

            if remote_nodemap == remote_nodemap2:
                print(node_name + '2 : ' + str(node_value.value))

    else:
        node1 = remote_nodemap.get_node(node_name1)

        if not node1.is_writable:
            print('not node'+ node_name1)
        else:
            if node1.principal_interface_type == st.EGCInterfaceType.IFloat:
                node_value1 = st.PyIFloat(node1)
            elif node1.principal_interface_type == st.EGCInterfaceType.IInteger:
                node_value1 = st.PyIInteger(node1)
            value1 = float(value)

            if node1.principal_interface_type == st.EGCInterfaceType.IFloat:
                value1 = float(value1)
            else:
                value1 = int(value1)
            if node_value1.min <= value1 <= node_value1.max:
                node_value1.value = value1

def set_balance_white_auto(index,value):
    enum_name1 = "BalanceRatioSelector"
    numeric_name1 = "BalanceRatio"
    node1 = remote_nodemap1.get_node(enum_name1)
    if not node1.is_writable:
        print('not node 1 '+ enum_name1)

    enum_node1 = st.PyIEnumeration(node1)
    enum_entries1 = enum_node1.entries

    enum_entry1 = enum_entries1[index]
    if enum_entry1.is_available:
        enum_node1.value = enum_entry1.value

        node_name2 = numeric_name1
        node2 = remote_nodemap1.get_node(node_name2)

        if not node2.is_writable:
            print('not node 2 '+ numeric_name1 + index)
        else:
            if node2.principal_interface_type == st.EGCInterfaceType.IFloat:
                node_value2 = st.PyIFloat(node2)
            elif node2.principal_interface_type == st.EGCInterfaceType.IInteger:
                node_value2 = st.PyIInteger(node2)
            value2 = value

            if node2.principal_interface_type == st.EGCInterfaceType.IFloat:
                value2 = float(value2)
            else:
                value2 = int(value2)
            if node_value2.min <= value2 <= node_value2.max:
                node_value2.value = value2
            print(st.PyIEnumEntry(enum_entry1).symbolic_value + ' : ' + str(node_value2.value))

def set_config_init_camera1():
    list_param_camera1 = []
    with open('static/param_camera1.txt') as lines:
        for line in lines:
            text, value = line.strip().split(':')
            list_param_camera1.append(value)
    set_exposure_or_gain(remote_nodemap1,"ExposureTime","ExposureTimeRaw",list_param_camera1[0])
    set_exposure_or_gain(remote_nodemap1,"Gain","GainRaw",list_param_camera1[1])
    set_balance_white_auto(0,list_param_camera1[2])
    set_balance_white_auto(3,list_param_camera1[3])
    set_balance_white_auto(4,list_param_camera1[4])


def set_config_init_camera2():
    list_param_camera2 = []
    with open('static/param_camera2.txt') as lines:
        for line in lines:
            _, value = line.strip().split(':')
            list_param_camera2.append(value)
    set_exposure_or_gain(remote_nodemap2,"ExposureTime","ExposureTimeRaw",list_param_camera2[0])
    set_exposure_or_gain(remote_nodemap2,"Gain","GainRaw",list_param_camera2[1])
    # set_balance_white_auto(0,list_param_camera2[2])
    # set_balance_white_auto(1,list_param_camera2[3])
    # set_balance_white_auto(2,list_param_camera2[4])
    # set_balance_white_auto(3,list_param_camera2[5])
    # set_balance_white_auto(4,list_param_camera2[6])

def load_param_camera1():
    list_param_camera1 = []
    with open('static/param_camera1.txt') as lines:
        for line in lines:
            _,value = line.strip().split(':')
            list_param_camera1.append(value)
    return list_param_camera1

def load_param_camera2():
    list_param_camera2 = []
    with open('static/param_camera2.txt') as lines:
        for line in lines:
            _,value = line.strip().split(':')
            list_param_camera2.append(value)
    return list_param_camera2

def save_param_camera1(v1,v2,v3,v4,v5):
    line1 = 'exposure1:' + str(v1)
    line2 = 'gain1:' + str(v2)
    line3 = 'red1:' + str(v3)
    line4 = 'green1:' + str(v4)
    line5 = 'blue1:' + str(v5)

    lines = [line1,line2,line3,line4,line5]
    with open('static/param_camera1.txt', "w") as f:
        for i in lines:
            f.write(i)
            f.write('\n')

def save_param_camera2(v1,v2):
    line1 = 'exposure2:' + str(v1)
    line2 = 'gain2:' + str(v2)
    lines = [line1,line2]
    with open('static/param_camera2.txt', "w") as f:
        for i in lines:
            f.write(i)
            f.write('\n')



def task_camera1(model,size,conf):
    read_2000 = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD0\x00') # doc thanh ghi 2000
    #print('cam1')
    #print(read_2000[-1:])
    temp1 = 1
    if temp1==1 and read_2000 == b'\xc0\x00\x02\x00\x19\x00\x00\x01\x00`\x01\x01\x00@\x00%':  # gia tri 37

        print('CAM 1')
        t1 = time.time()
        
        img1_orgin = my_callback1.image
        img1_orgin = img1_orgin[50:530,70:710]
        img1_orgin = img1_orgin.copy()

        # ghi vao D2000 gia tri 0
        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD0\x00',b'\x00\x00',1)


        img1_convert = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB)

        result1 = model(img1_convert,size= size,conf = conf) 
        t2 = time.time() - t1
        print(t2) 
        #time.sleep(10)
        table1 = result1.pandas().xyxy[0]
        area_remove1 = []
        for item in range(len(table1.index)):
            width1 = table1['xmax'][item] - table1['xmin'][item]
            height1 = table1['ymax'][item] - table1['ymin'][item]
            area1 = width1*height1
            if table1['name'][item] == 'nut_me':
                if area1 < values['area_nutme1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'divat':
                if area1 < values['area_divat1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'me':
                if area1 < values['area_me1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'namchamcao':
                if height1 < values['y_namchamcao1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'tray_bac_truc':
                if area1 < values['area_traybactruc1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'di_vat_duoi':
                if area1 < values['area_divatduoi1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'kimcao':
                if height1 < values['ymin_kimnamcham1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)
                elif height1 > values['ymax_kimnamcham1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

        names1 = list(table1['name'])
        print(names1)

        len_ncc = 0
        for ncc in names1:
            if ncc == 'namchamcao':
                len_ncc +=1
        
        len_kimcao = 0
        for kimcao in names1:
            if kimcao == 'kimcao':
                len_kimcao += 1

        save_memorys1 = []

        if 'tray_bac_truc' in names1 or 'di_vat_duoi' in names1:
            save_memorys1.append(1000)
        if 'kimnamcham' not in names1 or len_kimcao !=2:
            save_memorys1.append(1002)
        if 'divat' in names1 or 'me' in names1 or 'nut_me' in names1 or 'nut' in names1 or len_ncc !=2:
            save_memorys1.append(1004)

        time.sleep(mysleep)

        for save_memory1 in save_memorys1:
            # bac_truc
            if save_memory1 == 1000: 
                # ghi vao D1000 gia tri 1 
                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xE8\x00',b'\x00\x01',1)
                # ghi vao D1006 (03EE) gia tri 2 => khong ok
                #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEE\x00',b'\x00\x02',1)
            # kim nam cham
            if save_memory1 == 1002:
                # ghi vao D1002 gia tri 1 
                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEA\x00',b'\x00\x01',1)
                # ghi vao D1006 (03EE) gia tri 2 => khong ok
                #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEE\x00',b'\x00\x02',1)

            #nam cham
            if save_memory1 == 1004:
                # ghi vao D1004 gia tri 1 
                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEC\x00',b'\x00\x01',1)
                # ghi vao D1006 gia tri 2 => khong ok
                #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEE\x00',b'\x00\x02',1)

            #OK
        if len(save_memorys1) == 0:
            # ghi vao D1006 gia tri 1 
            fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEE\x0B',b'\x00\x01',1)
            # ghi vao D1000 gia tri 2
            #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xE8\x00',b'\x00\x02',1)
            # ghi vao D1002 gia tri 2
            #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEA\x00',b'\x00\x02',1)
            # ghi vao D1004 gia tri 2
            #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEC\x00',b'\x00\x02',1)             

        #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD0\x00',b'\x00\x04',1)
        t2 = time.time() - t1
        print(t2) 
        time_cam1 = str(int(t2*1000)) + 'ms'
        window['time_cam1'].update(value= time_cam1, text_color='black') 


        show1 = np.squeeze(result1.render(area_remove1))
        show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

        if 'kimnamcham' not in names1 or len_ncc !=2 or len_kimcao !=2 \
        or 'divat' in names1 or 'me' in names1 or 'nut_me' in names1 or 'nut' in names1 \
        or 'tray_bac_truc' in names1 or 'di_vat_duoi' in names1:
            print('NG')
            cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
            window['result_cam1'].update(value= 'NG', text_color='red')
            name_folder_ng = time_to_name()
            cv2.imwrite('G:/result/Cam1/NG/' + name_folder_ng + '.jpg',img1_orgin)
            cv2.imwrite('G:/Windows/1/' + name_folder_ng + '.jpg',img1_orgin)
        else:
            print('OK')
            cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
            window['result_cam1'].update(value= 'OK', text_color='green')
            name_folder_ok = time_to_name()
            cv2.imwrite('G:/result/Cam1/OK/' + name_folder_ok  + '.jpg',img1_orgin)
    

        imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
        window['image1'].update(data= imgbytes1)
        temp1=0
        print('---------------------------------------------')



def task_camera2(model,size,conf):
    read_2002 = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD2\x00') # doc thanh ghi 2002
    #print('cam2')
    #print(read_2002[-1:])
    temp2 = 1
    if temp2 == 1 and read_2002 ==b'\xc0\x00\x02\x00\x19\x00\x00\x01\x00`\x01\x01\x00@\x00%':  # gia tri 37

        print('CAM 2')
        t1 = time.time()
        img2_orgin = my_callback2.image
        #img2_orgin = cv2.cvtColor(img2_orgin, cv2.COLOR_BGR2RGB)

        # ghi vao D2002 gia tri 0
        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD2\x00',b'\x00\x00',1)

        result2 = model(img2_orgin,size= size,conf = conf) 
        t2 = time.time() - t1
        print(t2) 

        table2 = result2.pandas().xyxy[0]

        area_remove2 = []
        for item in range(len(table2.index)):
            width2 = table2['xmax'][item] - table2['xmin'][item]
            height2 = table2['ymax'][item] - table2['ymin'][item]
            area2 = width2*height2
            if table2['name'][item] == 'nut_me':
                if area2 < values['area_nutme2_a22']: 
                    table2.drop(item, axis=0, inplace=True)
                    area_remove2.append(item)

            elif table2['name'][item] == 'divat':
                if area2 < values['area_divat2_a22']: 
                    table2.drop(item, axis=0, inplace=True)
                    area_remove2.append(item)

            elif table2['name'][item] == 'me':
                if area2 < values['area_me2_a22']: 
                    table2.drop(item, axis=0, inplace=True)
                    area_remove2.append(item)

        names2 = list(table2['name'])
        print(names2)

        save_memorys2 = []
        if 'namcham' not in names2 or 'divat' in names2 or 'me' in names2 or 'namchamcao' in names2 or 'nut_me' in names2 or 'nut' in names2:
            save_memorys2.append(1014)
        # thieu kimnamcham 1012 va bactruc 1010

        time.sleep(mysleep)

        for save_memory2 in save_memorys2:
            # bac_truc
            if save_memory2 == 1010: 
                # ghi vao D1010 gia tri 1 
                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF2\x00',b'\x00\x01',1)
                # ghi vao D1016 gia tri 2 => khong ok
                #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF8\x00',b'\x00\x02',1)
            # kim nam cham
            if save_memory2 == 1012:
                # ghi vao D1012 gia tri 1 
                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF4\x00',b'\x00\x01',1)
                # ghi vao D1016 gia tri 2 => khong ok
                #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF8\x00',b'\x00\x02',1)

            #nam cham
            if save_memory2 == 1014:
                # ghi vao D1014 gia tri 1 
                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF6\x00',b'\x00\x01',1)
                # ghi vao D1016 gia tri 2 => khong ok
                #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF8\x00',b'\x00\x02',1)

        #OK
        if len(save_memorys2) == 0:
            # ghi vao D1016 gia tri 1 
            fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF8\x0B',b'\x00\x01',1)
            # ghi vao D1010 gia tri 2
            #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF2\x00',b'\x00\x02',1)
            # ghi vao D1012 gia tri 2
            #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF4\x00',b'\x00\x02',1)
            # ghi vao D1014 gia tri 2
            #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF6\x00',b'\x00\x02',1)             

        #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD2\x00',b'\x00\x04',1)
        t2 = time.time() - t1
        print(t2) 
        time_cam2 = str(int(t2*1000)) + 'ms'
        window['time_cam2'].update(value= time_cam2, text_color='black') 

        show2 = np.squeeze(result2.render(area_remove2))
        show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

        if 'namcham' not in names2 \
        or 'divat' in names2 or 'me' in names2 or 'namchamcao' in names2 or 'nut_me' in names2 or 'nut' in names2:
            print('NG')
            cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
            window['result_cam2'].update(value= 'NG', text_color='red')  
            name_folder_ng = time_to_name()
            cv2.imwrite('G:/result/Cam2/NG/' + name_folder_ng + '.jpg',img2_orgin)     
            cv2.imwrite('G:/Windows/2/' + name_folder_ng + '.jpg',img2_orgin)    
        else:
            print('OK')
            cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
            window['result_cam2'].update(value= 'OK', text_color='green')
            name_folder_ok = time_to_name()
            cv2.imwrite('G:/result/Cam2/OK/' + name_folder_ok + '.jpg',img2_orgin)        


        imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
        window['image2'].update(data= imgbytes2)
        temp2 = 0
        print('---------------------------------------------')
        

def test_camera1(model,size,conf):
    read_2000 = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD0\x00') # doc thanh ghi 2000
    if read_2000 ==b'\xc0\x00\x02\x00\x19\x00\x00\x01\x00`\x01\x01\x00@\x00%':  # gia tri 37
        print('CAM 1')
        t1 = time.time()
        img1_orgin = my_callback1.image
        img1_orgin = img1_orgin[50:530,70:710]
        img1_orgin = img1_orgin.copy()
        img1_convert = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB)
        result1 = model(img1_convert,size= size,conf = conf) 
        #img1_orgin = cv2.resize(img1_orgin,(640,480))


        # ghi vao D1006 gia tri 1 
        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEE\x0B',b'\x00\x01',1)
        # ghi vao D1000 gia tri 2
        #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xE8\x00',b'\x00\x02',1)
        # ghi vao D1002 gia tri 2
        #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEA\x00',b'\x00\x02',1)
        # ghi vao D1004 gia tri 2
        #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEC\x00',b'\x00\x02',1)             

        # ghi vao D2000 gia tri 0
        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD0\x00',b'\x00\x00',1)


        show1 = np.squeeze(result1.render())
        show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)


        print('OK')
        cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
        window['result_cam1'].update(value= 'OK', text_color='green')
        
        name_folder_ok = time_to_name()
        cv2.imwrite('G:/result/Cam1/OK/' + name_folder_ok  + '.jpg',img1_orgin)


        imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
        window['image1'].update(data= imgbytes1)

        #print('---------------------------------------------')


        # name_folder_ok = time_to_name()
        # cv2.imwrite('G:/result/Cam1/OK/' + name_folder_ok  + '.jpg',img1_orgin)

        # img1_orgin = cv2.resize(img1_orgin,(image_width_display,480))
        # imgbytes1 = cv2.imencode('.png',img1_orgin)[1].tobytes()
        # window['image1'].update(data= imgbytes1)

        t2 = time.time() - t1
        print(t2) 

        print('---------------------------------------------')





def test_camera2():
    read_2002 = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD2\x00') # doc thanh ghi 2002
    if read_2002 ==b'\xc0\x00\x02\x00\x19\x00\x00\x01\x00`\x01\x01\x00@\x00%':  # gia tri 37
        print('CAM 2')
        t1 = time.time()
        img2_orgin = my_callback2.image


        # ghi vao D1016 gia tri 1 
        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF8\x0B',b'\x00\x01',1)
        # ghi vao D1010 gia tri 2
        #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF2\x00',b'\x00\x02',1)
        # ghi vao D1012 gia tri 2
        #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF4\x00',b'\x00\x02',1)
        # ghi vao D1014 gia tri 2
        #fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF6\x00',b'\x00\x02',1) 

        # ghi vao D2002 gia tri 0
        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD2\x00',b'\x00\x00',1)            
        
        name_folder_ok = time_to_name()
        cv2.imwrite('G:/result/Cam2/OK/' + name_folder_ok  + '.jpg',img2_orgin)        
        
        img2_orgin = cv2.resize(img2_orgin,(640,480))
        imgbytes2 = cv2.imencode('.png',img2_orgin)[1].tobytes()
        window['image2'].update(data= imgbytes2)

        t2 = time.time() - t1
        print(t2) 


        print('---------------------------------------------')
        



def task_camera1_snap(model,size,conf):
    if event =='Snap1': 
        t1 = time.time()
        img1_orgin = my_callback1.image                            # 0.0
        #img1_orgin = cv2.resize(img1_orgin,(640,480))
        #img1_orgin = Image.open(img1_orgin)
        #cv2.imshow('asd',img1_orgin)
        # name_folder_ok = time_to_name()
        # cv2.imwrite('G:/result/Cam1/test/' + name_folder_ok  + '.jpg',img1_orgin)
        img1_orgin = img1_orgin[50:530,70:710]
        img1_orgin = img1_orgin.copy()
        img1_orgin = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB)
        result1 = model(img1_orgin,size= size,conf = conf)             # 0.015
        table1 = result1.pandas().xyxy[0]
        area_remove1 = []

        for item in range(len(table1.index)):
            width1 = table1['xmax'][item] - table1['xmin'][item]
            height1 = table1['ymax'][item] - table1['ymin'][item]
            area1 = width1*height1
            if table1['name'][item] == 'nut_me':
                if area1 < values['area_nutme1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'divat':
                if area1 < values['area_divat1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'me':
                if area1 < values['area_me1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'namchamcao':
                if height1 < values['y_namchamcao1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'tray_bac_truc':
                if area1 < values['area_traybactruc1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'di_vat_duoi':
                if area1 < values['area_divatduoi1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)

            elif table1['name'][item] == 'kimcao':
                if height1 < values['ymin_kimnamcham1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)
                elif height1 > values['ymax_kimnamcham1_a22']: 
                    table1.drop(item, axis=0, inplace=True)
                    area_remove1.append(item)
                print(height1)
        
        names1 = list(table1['name'])
        print(names1)

        show1 = np.squeeze(result1.render(area_remove1))
        show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

        len_ncc = 0
        len_kimcao = 0
        for find_name1 in names1:
            if find_name1 == 'namchamcao':
                len_ncc +=1

            if find_name1 == 'kimcao':
                len_kimcao += 1


        if 'kimnamcham' not in names1 or len_ncc !=2 or len_kimcao !=2 \
        or 'divat' in names1 or 'me' in names1 or 'nut_me' in names1 or 'nut' in names1 \
        or 'tray_bac_truc' in names1 or 'di_vat_duoi' in names1:
            print('NG')
            cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
            window['result_cam1'].update(value= 'NG', text_color='red')
            name_folder_ng = time_to_name()
            cv2.imwrite('G:/result/Cam1/NG/' + name_folder_ng + '.jpg',img1_orgin)
            cv2.imwrite('G:/Windows/1/' + name_folder_ng + '.jpg',img1_orgin)

        else:
            print('OK')
            cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
            window['result_cam1'].update(value= 'OK', text_color='green')
            name_folder_ok = time_to_name()
            cv2.imwrite('G:/result/Cam1/OK/' + name_folder_ok  + '.jpg',img1_orgin)
           
        imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
        window['image1'].update(data= imgbytes1)

        t2 = time.time() - t1
        print(t2) 
    
        print('---------------------------------------------')



def task_camera2_snap(model,size,conf):
    if event =='Snap2': 
        t3 = time.time()
        img2_orgin = my_callback2.image

        result2 = model(img2_orgin,size= size,conf = conf) 

        table2 = result2.pandas().xyxy[0]

        area_remove2 = []
        for item in range(len(table2.index)):
            width2 = table2['xmax'][item] - table2['xmin'][item]
            height2 = table2['ymax'][item] - table2['ymin'][item]
            area2 = width2*height2
            if table2['name'][item] == 'nut_me':
                if area2 < values['area_nutme2_a22']: 
                    table2.drop(item, axis=0, inplace=True)
                    area_remove2.append(item)

            elif table2['name'][item] == 'divat':
                if area2 < values['area_divat2_a22']: 
                    table2.drop(item, axis=0, inplace=True)
                    area_remove2.append(item)

            elif table2['name'][item] == 'me':
                if area2 < values['area_me2_a22']: 
                    table2.drop(item, axis=0, inplace=True)
                    area_remove2.append(item)

        names2 = list(table2['name'])
        print(names2)

        save_memorys2 = []
        if 'namcham' not in names2 or 'divat' in names2 or 'me' in names2 or 'namchamcao' in names2 or 'nut_me' in names2 or 'nut' in names2:
            save_memorys2.append(1014)
        # thieu kimnamcham 1012 va bactruc 1010

        show2 = np.squeeze(result2.render(area_remove2))
        show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

        if 'namcham' not in names2 \
        or 'divat' in names2 or 'me' in names2 or 'namchamcao' in names2 or 'nut_me' in names2 or 'nut' in names2:
            print('NG')
            cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
            window['result_cam2'].update(value= 'NG', text_color='red')  
            name_folder_ng = time_to_name()
            cv2.imwrite('G:/result/Cam2/NG/' + name_folder_ng + '.jpg',img2_orgin)   
            cv2.imwrite('G:/Windows/2/' + name_folder_ng + '.jpg',img2_orgin)     

        else:
            print('OK')
            cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
            window['result_cam2'].update(value= 'OK', text_color='green')
            name_folder_ok = time_to_name()
            cv2.imwrite('G:/result/Cam2/OK/' + name_folder_ok + '.jpg',img2_orgin)


        imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
        window['image2'].update(data= imgbytes2)
                        
        t4 = time.time() - t3
        print(t4) 

        print('---------------------------------------------')



def make_window(theme):
    sg.theme(theme)

    #file_img = [("JPEG (*.jpg)",("*jpg","*.png"))]

    file_weights = [('Weights (*.pt)', ('*.pt'))]

    # menu = [['Application', ['Connect PLC','Interrupt Connect PLC','Exit']],
    #         ['Tool', ['Check Cam','Change Theme']],
    #         ['Help',['About']]]

    right_click_menu = [[], ['Exit','Administrator','Change Theme']]

    layout_main = [

        #[sg.MenubarCustom(menu, font='Helvetica',text_color='white',background_color='#404040',bar_text_color='white',bar_background_color='#404040',bar_font='Helvetica')],
        # [sg.Text('CAM 1', size =(34,1),justification='center' ,font= ('Helvetica',30),text_color='red' ,relief= sg.RELIEF_SUNKEN),
        # sg.Text('CAM 2', size =(34,1),justification='center' ,font= ('Helvetica',30),text_color='red' ,relief= sg.RELIEF_SUNKEN)],
        
        # sg.Frame('',[
        #     sg.Text('CAM 2',justification='center' ,font= ('Helvetica',30),text_color='red',expand_x=True),
        #     sg.Text('CAM 1',justification='center' ,font= ('Helvetica',30),text_color='red',expand_x=True),
        # ]),

        [
        sg.Text('CAM 2',justification='center' ,font= ('Helvetica',30),text_color='red',expand_x=True),
        sg.Text('CAM 1',justification='center' ,font= ('Helvetica',30),text_color='red',expand_x=True)],
        # sg.Frame('',[
        #     [sg.Text('CAM 2',justification='center' ,font= ('Helvetica',30),text_color='red',expand_x=True),
        #     sg.Text('CAM 1',justification='center' ,font= ('Helvetica',30),text_color='red',expand_x=True)],
        # ]),

        [

        # 2
        sg.Frame('',[
            #[sg.Image(filename='', size=(640,480),key='image1',background_color='black')],
            [sg.Image(filename='', size=(image_width_display,image_height_display),key='image2',background_color='black',expand_x=True, expand_y=True)],
            [sg.Frame('',
            [
                [sg.Text('',auto_size_text=True,font=('Helvetica',120), justification='center', key='result_cam2',expand_x=True, expand_y=True)],
                [sg.Text('',font=('Helvetica',30),justification='center', key='time_cam2',expand_x=True, expand_y=True)],
            ], vertical_alignment='top',size=(520,250),expand_x=True, expand_y=True),
            sg.Frame('',[
                #[sg.Text('')],
                [sg.Button('Webcam', size=(8,1),  font=('Helvetica',14),disabled=True,key= 'Webcam2',expand_x=True, expand_y=True,auto_size_button=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                [sg.Button('Stop', size=(8,1), font=('Helvetica',14),disabled=True  ,key='Stop2',expand_x=True, expand_y=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                #[sg.Button('Continue', size=(8,1),  font=('Helvetica',14),disabled=True ,key='Continue2')],
                #[sg.Text('')],
                [sg.Button('Snap', size=(8,1), font=('Helvetica',14),disabled=True  ,key='Snap2',expand_x=True, expand_y=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                [sg.Checkbox('Model',size=(6,1),font=('Helvetica',14), disabled=True, key='have_model2',expand_x=True, expand_y=True)]

                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT,expand_x=True, expand_y=True),

            sg.Frame('',[   
                [sg.Button('Change', size=(8,1), font=('Helvetica',14), disabled= True, key= 'Change2',expand_x=True, expand_y=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                [sg.Button('Pic', size=(8,1), font=('Helvetica',14),disabled=True,key='Pic2',expand_x=True, expand_y=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                [sg.Button('Detect', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'Detect2',expand_x=True, expand_y=True)],
                [sg.Text('',size=(4,2),expand_x=True, expand_y=True)],
                [sg.Text('',size=(4,1),expand_x=True, expand_y=True)],
                #[sg.Button('SaveData', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'SaveData2_a22')],

                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT,expand_x=True, expand_y=True),
            ]
        ], expand_x=True, expand_y= True),

        #1
        sg.Frame('',[
            #[sg.Image(filename='', size=(640,480),key='image1',background_color='black')],
            [sg.Image(filename='', size=(image_width_display,image_height_display),key='image1',background_color='black',expand_x=True, expand_y=True)],
            [sg.Frame('',
            [
                [sg.Text('',auto_size_text=True,font=('Helvetica',120), justification='center', key='result_cam1',expand_x=True, expand_y=True)],
                [sg.Text('',font=('Helvetica',30), justification='center', key='time_cam1',expand_x=True, expand_y=True)],
            ], vertical_alignment='top',size=(520,250),expand_x=True, expand_y=True),
            sg.Frame('',[
                #[sg.Text('')],
                [sg.Button('Webcam', size=(8,1),  font=('Helvetica',14),disabled=True ,key= 'Webcam1',expand_x=True, expand_y=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                [sg.Button('Stop', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'Stop1',expand_x=True, expand_y=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                #[sg.Button('Continue', size=(8,1),  font=('Helvetica',14), disabled=True, key= 'Continue1')],
                #[sg.Text('')],
                [sg.Button('Snap', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'Snap1',expand_x=True, expand_y=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                [sg.Checkbox('Model',size=(6,1),font=('Helvetica',14), disabled=True, key='have_model1',expand_x=True, expand_y=True)]
                #],element_justification='center',expand_x=True, vertical_alignment='top', relief= sg.RELIEF_FLAT),
                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT,expand_x=True, expand_y=True),
                
            sg.Frame('',[   
                [sg.Button('Change', size=(8,1), font=('Helvetica',14), disabled= True, key= 'Change1',expand_x=True, expand_y=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                [sg.Button('Pic', size=(8,1), font=('Helvetica',14),disabled=True,key= 'Pic1',expand_x=True, expand_y=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                [sg.Button('Detect', size=(8,1), font=('Helvetica',14),disabled=True,key= 'Detect1',expand_x=True, expand_y=True)],
                [sg.Text('',expand_x=True, expand_y=True)],
                [sg.InputCombo(('A22','A19'),size=(1,200),font=('Helvetica',20),default_value=choose_model,key='change_model',enable_events=True,expand_x=True, expand_y=True)],
                # [sg.Text('',size=(14,2),expand_x=True, expand_y=True)],
                # [sg.Text('',size=(14,1),expand_x=True, expand_y=True)],
                #[sg.Button('SaveData', size=(8,1), font=('Helvetica',14),disabled=True,key= 'SaveData1_a22')],

                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT,expand_x=True, expand_y=True),
            ],
                
        ],expand_x=True, expand_y=True),
    
    ]] 


    layout_parameter_model_A22 = [
        [sg.Text('CAM 2', size =(34,1),justification='center' ,font= ('Helvetica',30),text_color='red'),
         sg.Text('CAM 1', size =(34,1),justification='center' ,font= ('Helvetica',30),text_color='red')],
        # 2
        [
        sg.Frame('',[
            [sg.Frame('',
            [
                [sg.Text('Weights A22', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(52,1), font=('Helvetica',12), key='file_weights2_a22',readonly= True, text_color='navy',default_text=list_param_model2_a22[0],enable_events= True),
                sg.Frame('',[
                    [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse2_a22',enable_events=True, disabled=True)]
                ], relief= sg.RELIEF_FLAT)],

                [sg.Text('Size', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(56,20),font=('Helvetica',11),disabled=True,default_value=list_param_model2_a22[1],key='imgsz2_a22')],
                [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(52,20),font=('Helvetica',11),disabled=True,default_value=list_param_model2_a22[2], key= 'conf_thres2_a22')],
                [sg.Text('Nut me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11),disabled=True,default_value=list_param_model2_a22[3], resolution=5, key= 'area_nutme2_a22')],
                [sg.Text('Di vat',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11),disabled=True,default_value=list_param_model2_a22[4], resolution=5, key= 'area_divat2_a22')],
                [sg.Text('Me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11),disabled=True ,default_value=list_param_model2_a22[5], resolution=5, key= 'area_me2_a22')],
                [sg.Text(' ')],
                [sg.Text(' '*152), sg.Button('Save Data', size=(12,1),  font=('Helvetica',12),key='SaveData2_a22',enable_events=True)] 
            ]),
            ]
        ], expand_y= True),

        #1
        sg.Frame('',[
            [sg.Frame('',
            [
                [sg.Text('Weights A22', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(52,1), font=('Helvetica',12), key='file_weights1_a22',readonly= True, text_color='navy',default_text=list_param_model1_a22[0],enable_events= True),
                sg.Frame('',[
                    [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse1_a22',enable_events=True,disabled=True )]
                ], relief= sg.RELIEF_FLAT)],

                [sg.Text('Size', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(56,20),font=('Helvetica',11),disabled=True ,default_value=list_param_model1_a22[1],key='imgsz1_a22')],
                [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(52,20),font=('Helvetica',11),disabled=True  ,default_value=list_param_model1_a22[2], key= 'conf_thres1_a22')],
                [sg.Text('Nut me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a22[3], resolution=5, key= 'area_nutme1_a22')],
                [sg.Text('Di vat',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a22[4], resolution=5, key= 'area_divat1_a22')],
                [sg.Text('Me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a22[5], resolution=5, key= 'area_me1_a22')],
                [sg.Text('Nam cham cao',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,500),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a22[6], resolution=1, key= 'y_namchamcao1_a22')],
                [sg.Text('Tray bac truc',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a22[7], resolution=5, key= 'area_traybactruc1_a22')],
                [sg.Text('Di vat duoi',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a22[8], resolution=5, key= 'area_divatduoi1_a22')],
                [sg.Text('Kim cao min',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,500),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a22[9], resolution=1, key= 'ymin_kimnamcham1_a22')],
                [sg.Text('Kim cao max',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,500),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a22[10], resolution=1, key= 'ymax_kimnamcham1_a22')],
                [sg.Text(' ')],
                [sg.Text(' '*152), sg.Button('Save Data', size=(12,1),  font=('Helvetica',12),key='SaveData1_a22',enable_events=True)] 
            ]),
            ]
        ], expand_y= True),
    
    ]] 


    layout_parameter_model_A19 = [
        [sg.Text('CAM 2', size =(34,1),justification='center' ,font= ('Helvetica',30),text_color='red'),
         sg.Text('CAM 1', size =(34,1),justification='center' ,font= ('Helvetica',30),text_color='red')],
        # 2
        [
        sg.Frame('',[
            [sg.Frame('',
            [

                [sg.Text('Weights A19', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(52,1), font=('Helvetica',12), key='file_weights2_a19',readonly= True, text_color='navy',default_text=list_param_model2_a19[0],enable_events= True),
                sg.Frame('',[
                    [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse2_a19',enable_events=True, disabled=True)]
                ], relief= sg.RELIEF_FLAT)],
                [sg.Text('Size', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(56,20),font=('Helvetica',11),disabled=True,default_value=list_param_model2_a19[1],key='imgsz2_a19')],
                [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(52,20),font=('Helvetica',11),disabled=True,default_value=list_param_model2_a19[2], key= 'conf_thres2_a19')],
                [sg.Text('Nut me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11),disabled=True,default_value=list_param_model2_a19[3], resolution=5, key= 'area_nutme2_a19')],
                [sg.Text('Di vat',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11),disabled=True,default_value=list_param_model2_a19[4], resolution=5, key= 'area_divat2_a19')],
                [sg.Text('Me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11),disabled=True ,default_value=list_param_model2_a19[5], resolution=5, key= 'area_me2_a19')],
                [sg.Text(' ')],
                [sg.Text(' '*152), sg.Button('Save Data', size=(12,1),  font=('Helvetica',12),key='SaveData2_a19',enable_events=True)] 
            ]),
            ]
        ], expand_y= True),

        #1
        sg.Frame('',[
            [sg.Frame('',
            [

                [sg.Text('Weights A19', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(52,1), font=('Helvetica',12), key='file_weights1_a19',readonly= True, text_color='navy',default_text=list_param_model1_a19[0],enable_events= True),
                sg.Frame('',[
                    [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse1_a19',enable_events=True,disabled=True )]
                ], relief= sg.RELIEF_FLAT)],
                [sg.Text('Size', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(56,20),font=('Helvetica',11),disabled=True ,default_value=list_param_model1_a19[1],key='imgsz1_a19')],
                [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(52,20),font=('Helvetica',11),disabled=True  ,default_value=list_param_model1_a19[2], key= 'conf_thres1_a19')],
                [sg.Text('Nut me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a19[3], resolution=5, key= 'area_nutme1_a19')],
                [sg.Text('Di vat',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a19[4], resolution=5, key= 'area_divat1_a19')],
                [sg.Text('Me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a19[5], resolution=5, key= 'area_me1_a19')],
                [sg.Text('Nam cham cao',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,500),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a19[6], resolution=1, key= 'y_namchamcao1_a19')],
                [sg.Text('Tray bac truc',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a19[7], resolution=5, key= 'area_traybactruc1_a19')],
                [sg.Text('Di vat duoi',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,20000),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a19[8], resolution=5, key= 'area_divatduoi1_a19')],
                [sg.Text('Kim cao min',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,500),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a19[9], resolution=1, key= 'ymin_kimnamcham1_a19')],
                [sg.Text('Kim cao max',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,500),orientation='h',size=(52,20),font=('Helvetica',11), disabled=True ,default_value=list_param_model1_a19[10], resolution=1, key= 'ymax_kimnamcham1_a19')],
                [sg.Text(' ')],
                [sg.Text(' '*152), sg.Button('Save Data', size=(12,1),  font=('Helvetica',12),key='SaveData1_a19',enable_events=True)] 
            ]),
            ]
        ], expand_y= True),
    
    ]] 


    layout_terminal = [[sg.Text("Anything printed will display here!")],
                      [sg.Multiline( font=('Helvetica',14), expand_x=True, expand_y=True, write_only=True, autoscroll=True, auto_refresh=True,reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True)]
                      ]
    
    layout = [[sg.TabGroup([[  sg.Tab('Main', layout_main),
                               sg.Tab('Parameter Model A22', layout_parameter_model_A22),
                               sg.Tab('Parameter Model A19', layout_parameter_model_A19),
                               sg.Tab('Output', layout_terminal)]], expand_x=True, expand_y=True)
               ]]

    layout[-1].append(sg.Sizegrip())
    window = sg.Window('HuynhLeVu', layout, location=(0,0),right_click_menu=right_click_menu,resizable=True).Finalize()
    window.bind('<Configure>',"Configure")
    window.Maximize()

    return window

image_width_display = 760
image_height_display = 480

result_width_display = 570
# image_width_display - 190
result_height_display = 100 


file_name_img = [("Img(*.jpg,*.png)",("*jpg","*.png"))]

recording1 = False
recording2 = False 

error_cam1 = True
error_cam2 = True

choose_model = load_choose_model()


list_param_model1_a22 = load_param_model('static/param_model1_A22.txt')
list_param_model2_a22 = load_param_model('static/param_model2_A22.txt')
list_param_model1_a19 = load_param_model('static/param_model1_A19.txt')
list_param_model2_a19 = load_param_model('static/param_model2_A19.txt')


list_param_camera1 = load_param_camera1()
list_param_camera2 = load_param_camera2()

themes = load_theme()
theme = themes[0]
window = make_window(theme)

window['result_cam1'].update(value= 'Wait', text_color='yellow')
window['result_cam2'].update(value= 'Wait', text_color='yellow')




# connected = False
# while connected == False:
#     connected = connect_plc('192.168.250.1')
#     print('connecting ....')
#     #event, values = window.read(timeout=20)

# print("connected plc")   


# try:
#     my_callback1 = CMyCallback()
#     cb_func1 = my_callback1.datastream_callback1

#     my_callback2 = CMyCallback()
#     cb_func2 = my_callback2.datastream_callback2

#     st.initialize()
#     st_system = st.create_system()


# except Exception as exception:
#     print('Error: ',exception)




# st_datastream2, st_device2, remote_nodemap2= setup_camera2_stc()

# st_datastream1, st_device1, remote_nodemap1= setup_camera1_stc()
# #st_datastream2, st_device2, remote_nodemap2= setup_camera2_stc()

# config_off_auto1(remote_nodemap1)
# set_config_init_camera1()

# config_off_auto2(remote_nodemap2)
# set_config_init_camera2()





#mypath1 = "G:/Connect/4/fins_omron/fins/best2_h.pt"
#mypath1 = os.path.join(os.getcwd(),'model/' + list_param_model1_a22[0])
if choose_model == 'A22':
    mypath1 = list_param_model1_a22[0]
    model1 = torch.hub.load('./levu','custom', path= mypath1, source='local',force_reload =False)
elif choose_model == 'A19':
    mypath1 = list_param_model1_a19[1]
    model1 = torch.hub.load('./levu','custom', path= mypath1, source='local',force_reload =False)

img1_test = os.path.join(os.getcwd(), 'img/imgtest.jpg')
result1 = model1(img1_test,416,0.25) 
print('model1 already')

#mypath2 = "G:/Connect/4/fins_omron/fins/best2_v.pt"
#mypath2 = os.path.join(os.getcwd(),'model/' + list_param_model2_a22[0])
if choose_model == 'A22':
    mypath2 = list_param_model2_a22[0]
    model2 = torch.hub.load('./levu','custom', path= mypath2, source='local',force_reload =False)
elif choose_model == 'A19':
    mypath2 = list_param_model2_a19[1]
    model2 = torch.hub.load('./levu','custom', path= mypath2, source='local',force_reload =False)


img2_test = os.path.join(os.getcwd(), 'img/imgtest.jpg')
result2 = model2(img2_test,416,0.25) 
print('model2 already')

window['result_cam1'].update(value= 'Done', text_color='blue')
window['result_cam2'].update(value= 'Done', text_color='blue')

#removefile()
try:
    while True:
        event, values = window.read(timeout=20)

#         if event =='Exit' or event == sg.WINDOW_CLOSED :
#             break

#         if event == 'Configure':
#             if window.TKroot.state() == 'zoomed':
#                 #print(window['image1'].get_size()[0])
#                 image_width_display = window['image1'].get_size()[0]
#                 image_height_display = window['image1'].get_size()[1]
#                 result_width_display = image_width_display - 190
#                 result_height_display = 100 


#         if event =='Administrator':
#             login_password = 'vu123'  # helloworld
#             password = sg.popup_get_text(
#                 'Enter Password: ', password_char='*') 
#             if password == login_password:
#                 sg.popup_ok('Login Successed!!! ',text_color='green', font=('Helvetica',14))  

#                 window['imgsz2_a22'].update(disabled= False)
#                 window['imgsz1_a22'].update(disabled= False)
#                 window['conf_thres2_a22'].update(disabled= False)
#                 window['conf_thres1_a22'].update(disabled= False)
#                 window['area_nutme2_a22'].update(disabled= False)
#                 window['area_nutme1_a22'].update(disabled= False)
#                 window['area_divat2_a22'].update(disabled= False)
#                 window['area_divat1_a22'].update(disabled= False)
#                 window['area_me2_a22'].update(disabled= False)
#                 window['area_me1_a22'].update(disabled= False)
#                 window['y_namchamcao1_a22'].update(disabled= False)
#                 window['area_traybactruc1_a22'].update(disabled= False)
#                 window['area_divatduoi1_a22'].update(disabled= False)
#                 window['ymin_kimnamcham1_a22'].update(disabled= False)
#                 window['ymax_kimnamcham1_a22'].update(disabled= False)

#                 window['imgsz2_a19'].update(disabled= False)
#                 window['imgsz1_a19'].update(disabled= False)
#                 window['conf_thres2_a19'].update(disabled= False)
#                 window['conf_thres1_a19'].update(disabled= False)
#                 window['area_nutme2_a19'].update(disabled= False)
#                 window['area_nutme1_a19'].update(disabled= False)
#                 window['area_divat2_a19'].update(disabled= False)
#                 window['area_divat1_a19'].update(disabled= False)
#                 window['area_me2_a19'].update(disabled= False)
#                 window['area_me1_a19'].update(disabled= False)
#                 window['y_namchamcao1_a19'].update(disabled= False)
#                 window['area_traybactruc1_a19'].update(disabled= False)
#                 window['area_divatduoi1_a19'].update(disabled= False)
#                 window['ymin_kimnamcham1_a19'].update(disabled= False)
#                 window['ymax_kimnamcham1_a19'].update(disabled= False)

#                 window['file_browse2_a22'].update(disabled= False,button_color='turquoise')
#                 window['file_browse1_a22'].update(disabled= False,button_color='turquoise')

#                 window['file_browse2_a19'].update(disabled= False,button_color='turquoise')
#                 window['file_browse1_a19'].update(disabled= False,button_color='turquoise')

#                 window['SaveData1_a22'].update(disabled= False,button_color='turquoise')
#                 window['SaveData2_a22'].update(disabled= False,button_color='turquoise')

#                 window['SaveData1_a19'].update(disabled= False,button_color='turquoise')
#                 window['SaveData2_a19'].update(disabled= False,button_color='turquoise')

#                 window['Webcam1'].update(disabled= False,button_color='turquoise')
#                 window['Webcam2'].update(disabled= False,button_color='turquoise')
#                 window['Stop1'].update(disabled= False,button_color='turquoise')
#                 window['Stop2'].update(disabled= False,button_color='turquoise')
#                 window['Pic1'].update(disabled= False,button_color='turquoise')
#                 window['Pic2'].update(disabled= False,button_color='turquoise')
#                 window['Snap1'].update(disabled= False,button_color='turquoise')
#                 window['Snap2'].update(disabled= False,button_color='turquoise')
#                 window['Change1'].update(button_color='turquoise')
#                 window['Change2'].update(button_color='turquoise')
#                 window['Detect1'].update(button_color='turquoise')
#                 window['Detect2'].update(button_color='turquoise')

#                 window['exposure_input1'].update(disabled=False)
#                 window['exposure_slider1'].update(disabled=False)
#                 window['choose_exposure1'].update(disabled=False,button_color='turquoise')
#                 window['gain_input1'].update(disabled=False)
#                 window['gain_slider1'].update(disabled=False)
#                 window['choose_gain1'].update(disabled=False,button_color='turquoise')
#                 window['aplly_param_camera1'].update(disabled=False,button_color='turquoise')
#                 window['save_param_camera1'].update(disabled=False,button_color='turquoise')
#                 window['have_model1'].update(disabled=False)

#                 window['exposure_input2'].update(disabled=False)
#                 window['exposure_slider2'].update(disabled=False)
#                 window['choose_exposure2'].update(disabled=False,button_color='turquoise')
#                 window['gain_input2'].update(disabled=False)
#                 window['gain_slider2'].update(disabled=False)
#                 window['choose_gain2'].update(disabled=False,button_color='turquoise')
#                 window['aplly_param_camera2'].update(disabled=False,button_color='turquoise')
#                 window['save_param_camera2'].update(disabled=False,button_color='turquoise')
#                 window['have_model2'].update(disabled=False)

#                 window['red_input1'].update(disabled=False)
#                 window['red_slider1'].update(disabled=False)        
#                 window['choose_red1'].update(disabled=False,button_color='turquoise')

#                 # window['greenr_input1'].update(disabled=False)    
#                 # window['greenr_slider1'].update(disabled=False)
#                 # window['choose_greenr1'].update(disabled=False,button_color='turquoise')   

#                 # window['greenb_input1'].update(disabled=False)    
#                 # window['greenb_slider1'].update(disabled=False)
#                 # window['choose_greenb1'].update(disabled=False,button_color='turquoise')   

#                 window['green_input1'].update(disabled=False)    
#                 window['green_slider1'].update(disabled=False)
#                 window['choose_green1'].update(disabled=False,button_color='turquoise')   

#                 window['blue_input1'].update(disabled=False)    
#                 window['blue_slider1'].update(disabled=False)
#                 window['choose_blue1'].update(disabled=False,button_color='turquoise')   


    
#             else:
#                 sg.popup_cancel('Wrong Password!!!',text_color='red', font=('Helvetica',14))

#         if event == 'Change Theme':
#             layout_theme = [
#                 [sg.Listbox(values= sg.theme_list(), size = (30,20),auto_size_text=18,default_values='Dark',key='theme', enable_events=True)],
#                 [
#                     [sg.Button('Apply'),
#                     sg.Button('Cancel')]
#                 ]
#             ] 
#             window_theme = sg.Window('Change Theme', layout_theme, location=(50,50),keep_on_top=True).Finalize()
#             window_theme.set_min_size((300,400))   

#             while True:
#                 event_theme, values_theme = window_theme.read(timeout=20)
#                 if event_theme == sg.WIN_CLOSEG:
#                     break

#                 if event_theme == 'Apply':
#                     theme_choose = values_theme['theme'][0]
#                     if theme_choose == 'Default':
#                         continue
#                     window.close()
#                     window = make_window(theme_choose)
#                     save_theme(theme_choose)
#                     #print(theme_choose)
#                 if event_theme == 'Cancel':
#                     answer = sg.popup_yes_no('Do you want to exit?')
#                     if answer == 'Yes':
#                         break
#                     if answer == 'No':
#                         continue
#             window_theme.close()


#         if event == 'file_browse1_a22': 
#             window['file_weights1_a22'].update(value=values['file_browse1_a22'])
#             if values['file_browse1_a22']:
#                 window['Change1'].update(disabled=False)
#         if event == 'file_browse1_a19': 
#             window['file_weights1_a19'].update(value=values['file_browse1_a19'])
#             if values['file_browse1_a19']:
#                 window['Change1'].update(disabled=False)

#         if event == 'file_browse2_a22':
#             window['file_weights2_a22'].update(value=values['file_browse2_a22'])
#             if values['file_browse2_a22']:
#                 window['Change2'].update(disabled=False)

#         if event == 'file_browse2_a19':
#             window['file_weights2_a19'].update(value=values['file_browse2_a19'])
#             if values['file_browse2_a19']:
#                 window['Change2'].update(disabled=False)

# #change model 
#         if event == 'change_model':
#             if values['change_model'] == 'A22':
#                 model1 = torch.hub.load('./levu','custom', path= list_param_model1_a22[0], source='local',force_reload =False)
#                 model2 = torch.hub.load('./levu','custom', path= list_param_model2_a22[0], source='local',force_reload =False)
#                 save_choose_model(values['change_model'])

#             if values['change_model'] == 'A19':
#                 model1 = torch.hub.load('./levu','custom', path= list_param_model1_a19[1], source='local',force_reload =False)
#                 model2 = torch.hub.load('./levu','custom', path= list_param_model2_a19[1], source='local',force_reload =False)
#                 save_choose_model(values['change_model'])


#         if event == 'SaveData1_a22':
#             save_param_model1_A22(values['file_weights1_a22'], values['imgsz1_a22'],values['conf_thres1_a22'],values['area_nutme1_a22'], values['area_divat1_a22'],values['area_me1_a22'],values['y_namchamcao1_a22'],values['area_traybactruc1_a22'],values['area_divatduoi1_a22'] ,values['ymin_kimnamcham1_a22'],values['ymax_kimnamcham1_a22'])
#             sg.popup('Saved param model 1 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


#         if event == 'SaveData2_a22':
#             save_param_model2_A22(values['file_weights2_a22'],values['imgsz2_a22'],values['conf_thres2_a22'] ,values['area_nutme2_a22'], values['area_divat2_a22'],values['area_me2_a22'])
#             sg.popup('Saved param model 2 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


#         if event == 'SaveData1_a19':
#             save_param_model1_A19(values['file_weights1_a19'], values['imgsz1_a19'],values['conf_thres1_a19'],values['area_nutme1_a19'], values['area_divat1_a19'],values['area_me1_a19'],values['y_namchamcao1_a19'],values['area_traybactruc1_a19'],values['area_divatduoi1_a19'] ,values['ymin_kimnamcham1_a19'],values['ymax_kimnamcham1_a19'])
#             sg.popup('Saved param model 1 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


#         if event == 'SaveData2_a19':
#             save_param_model2_A19(values['file_weights2_a19'], values['imgsz2_a19'],values['conf_thres2_a19'] ,values['area_nutme2_a19'], values['area_divat2_a19'],values['area_me2_a19'])
#             sg.popup('Saved param model 2 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)




        #print(cb_func1)
        #print(cb_func2)
        if values['change_model'] == 'A22':

            if event =='Exit' or event == sg.WINDOW_CLOSED :
                break

            if event == 'Configure':
                if window.TKroot.state() == 'zoomed':
                    #print(window['image1'].get_size()[0])
                    image_width_display = window['image1'].get_size()[0]
                    image_height_display = window['image1'].get_size()[1]
                    result_width_display = image_width_display - 190
                    result_height_display = 100 


            if event =='Administrator':
                login_password = 'vu123'  # helloworld
                password = sg.popup_get_text(
                    'Enter Password: ', password_char='*') 
                if password == login_password:
                    sg.popup_ok('Login Successed!!! ',text_color='green', font=('Helvetica',14))  

                    window['imgsz2_a22'].update(disabled= False)
                    window['imgsz1_a22'].update(disabled= False)
                    window['conf_thres2_a22'].update(disabled= False)
                    window['conf_thres1_a22'].update(disabled= False)
                    window['area_nutme2_a22'].update(disabled= False)
                    window['area_nutme1_a22'].update(disabled= False)
                    window['area_divat2_a22'].update(disabled= False)
                    window['area_divat1_a22'].update(disabled= False)
                    window['area_me2_a22'].update(disabled= False)
                    window['area_me1_a22'].update(disabled= False)
                    window['y_namchamcao1_a22'].update(disabled= False)
                    window['area_traybactruc1_a22'].update(disabled= False)
                    window['area_divatduoi1_a22'].update(disabled= False)
                    window['ymin_kimnamcham1_a22'].update(disabled= False)
                    window['ymax_kimnamcham1_a22'].update(disabled= False)

                    window['imgsz2_a19'].update(disabled= False)
                    window['imgsz1_a19'].update(disabled= False)
                    window['conf_thres2_a19'].update(disabled= False)
                    window['conf_thres1_a19'].update(disabled= False)
                    window['area_nutme2_a19'].update(disabled= False)
                    window['area_nutme1_a19'].update(disabled= False)
                    window['area_divat2_a19'].update(disabled= False)
                    window['area_divat1_a19'].update(disabled= False)
                    window['area_me2_a19'].update(disabled= False)
                    window['area_me1_a19'].update(disabled= False)
                    window['y_namchamcao1_a19'].update(disabled= False)
                    window['area_traybactruc1_a19'].update(disabled= False)
                    window['area_divatduoi1_a19'].update(disabled= False)
                    window['ymin_kimnamcham1_a19'].update(disabled= False)
                    window['ymax_kimnamcham1_a19'].update(disabled= False)

                    window['file_browse2_a22'].update(disabled= False,button_color='turquoise')
                    window['file_browse1_a22'].update(disabled= False,button_color='turquoise')

                    window['file_browse2_a19'].update(disabled= False,button_color='turquoise')
                    window['file_browse1_a19'].update(disabled= False,button_color='turquoise')

                    window['SaveData1_a22'].update(disabled= False,button_color='turquoise')
                    window['SaveData2_a22'].update(disabled= False,button_color='turquoise')

                    window['SaveData1_a19'].update(disabled= False,button_color='turquoise')
                    window['SaveData2_a19'].update(disabled= False,button_color='turquoise')

                    window['Webcam1'].update(disabled= False,button_color='turquoise')
                    window['Webcam2'].update(disabled= False,button_color='turquoise')
                    window['Stop1'].update(disabled= False,button_color='turquoise')
                    window['Stop2'].update(disabled= False,button_color='turquoise')
                    window['Pic1'].update(disabled= False,button_color='turquoise')
                    window['Pic2'].update(disabled= False,button_color='turquoise')
                    window['Snap1'].update(disabled= False,button_color='turquoise')
                    window['Snap2'].update(disabled= False,button_color='turquoise')
                    window['Change1'].update(button_color='turquoise')
                    window['Change2'].update(button_color='turquoise')
                    window['Detect1'].update(button_color='turquoise')
                    window['Detect2'].update(button_color='turquoise')

                    window['have_model1'].update(disabled=False)


                    window['have_model2'].update(disabled=False)


        
                else:
                    sg.popup_cancel('Wrong Password!!!',text_color='red', font=('Helvetica',14))

            if event == 'Change Theme':
                layout_theme = [
                    [sg.Listbox(values= sg.theme_list(), size = (30,20),auto_size_text=18,default_values='Dark',key='theme', enable_events=True)],
                    [
                        [sg.Button('Apply'),
                        sg.Button('Cancel')]
                    ]
                ] 
                window_theme = sg.Window('Change Theme', layout_theme, location=(50,50),keep_on_top=True).Finalize()
                window_theme.set_min_size((300,400))   

                while True:
                    event_theme, values_theme = window_theme.read(timeout=20)
                    if event_theme == sg.WIN_CLOSEG:
                        break

                    if event_theme == 'Apply':
                        theme_choose = values_theme['theme'][0]
                        if theme_choose == 'Default':
                            continue
                        window.close()
                        window = make_window(theme_choose)
                        save_theme(theme_choose)
                        #print(theme_choose)
                    if event_theme == 'Cancel':
                        answer = sg.popup_yes_no('Do you want to exit?')
                        if answer == 'Yes':
                            break
                        if answer == 'No':
                            continue
                window_theme.close()


            if event == 'file_browse1_a22': 
                window['file_weights1_a22'].update(value=values['file_browse1_a22'])
                if values['file_browse1_a22']:
                    window['Change1'].update(disabled=False)
            if event == 'file_browse1_a19': 
                window['file_weights1_a19'].update(value=values['file_browse1_a19'])
                if values['file_browse1_a19']:
                    window['Change1'].update(disabled=False)

            if event == 'file_browse2_a22':
                window['file_weights2_a22'].update(value=values['file_browse2_a22'])
                if values['file_browse2_a22']:
                    window['Change2'].update(disabled=False)

            if event == 'file_browse2_a19':
                window['file_weights2_a19'].update(value=values['file_browse2_a19'])
                if values['file_browse2_a19']:
                    window['Change2'].update(disabled=False)

    #change model 
            if event == 'change_model':
                if values['change_model'] == 'A22':
                    model1 = torch.hub.load('./levu','custom', path= list_param_model1_a22[0], source='local',force_reload =False)
                    model2 = torch.hub.load('./levu','custom', path= list_param_model2_a22[0], source='local',force_reload =False)
                    save_choose_model(values['change_model'])
                    window['result_cam1'].update(value='A22')
                    window['result_cam2'].update(value='A22')

                if values['change_model'] == 'A19':
                    model1 = torch.hub.load('./levu','custom', path= list_param_model1_a19[0], source='local',force_reload =False)
                    model2 = torch.hub.load('./levu','custom', path= list_param_model2_a19[0], source='local',force_reload =False)
                    save_choose_model(values['change_model'])
                    window['result_cam1'].update(value='A19')
                    window['result_cam2'].update(value='A19')


            if event == 'SaveData1_a22':
                save_param_model1_A22(values['file_weights1_a22'], values['imgsz1_a22'],values['conf_thres1_a22'],values['area_nutme1_a22'], values['area_divat1_a22'],values['area_me1_a22'],values['y_namchamcao1_a22'],values['area_traybactruc1_a22'],values['area_divatduoi1_a22'] ,values['ymin_kimnamcham1_a22'],values['ymax_kimnamcham1_a22'])
                sg.popup('Saved param model 1 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


            if event == 'SaveData2_a22':
                save_param_model2_A22(values['file_weights2_a22'],values['imgsz2_a22'],values['conf_thres2_a22'] ,values['area_nutme2_a22'], values['area_divat2_a22'],values['area_me2_a22'])
                sg.popup('Saved param model 2 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


            if event == 'SaveData1_a19':
                save_param_model1_A19(values['file_weights1_a19'], values['imgsz1_a19'],values['conf_thres1_a19'],values['area_nutme1_a19'], values['area_divat1_a19'],values['area_me1_a19'],values['y_namchamcao1_a19'],values['area_traybactruc1_a19'],values['area_divatduoi1_a19'] ,values['ymin_kimnamcham1_a19'],values['ymax_kimnamcham1_a19'])
                sg.popup('Saved param model 1 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


            if event == 'SaveData2_a19':
                save_param_model2_A19(values['file_weights2_a19'], values['imgsz2_a19'],values['conf_thres2_a19'] ,values['area_nutme2_a19'], values['area_divat2_a19'],values['area_me2_a19'])
                sg.popup('Saved param model 2 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


                

            task_camera1_snap(model=model1,size= values['imgsz1_a22'],conf= values['conf_thres1_a22']/100)
            task_camera2_snap(model=model2,size= values['imgsz2_a22'],conf= values['conf_thres2_a22']/100)

            #task_camera1(model=model1,size= values['imgsz1_a22'],conf= values['conf_thres1_a22']/100)
            #task_camera2(model=model2,size= values['imgsz2_a22'],conf= values['conf_thres2_a22']/100)




            #test_camera1(model=model1,size= values['imgsz1_a22'],conf= values['conf_thres1_a22']/100)
            #test_camera2()

            #task1(model1,size= values['imgsz1_a22'],conf= values['conf_thres1_a22']/100)
            #task2(model2,size= values['imgsz2_a22'],conf= values['conf_thres2_a22']/100) 

            #task1(model,size,conf)
            #task2(model,size,conf) 


            ### threading

            #task1 = threading.Thread(target=task_camera1, args=(model1, values['imgsz1_a22'], values['conf_thres1_a22']/100,))
            #task2 = threading.Thread(target=task_camera2, args=(model2, values['imgsz2_a22'], values['conf_thres2_a22']/100,))

            #task1.start()
            #task2.start()

            #task1.join()
            #task2.join()

            # menu



            if event == 'Webcam1':
                #cap1 = cv2.VideoCapture(0)
                recording1 = True


            elif event == 'Stop1':
                recording1 = False 
                imgbytes1 = np.zeros([100,100,3],dtype=np.uint8)
                imgbytes1 = cv2.resize(imgbytes1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                imgbytes1 = cv2.imencode('.png',imgbytes1)[1].tobytes()
                window['image1'].update(data=imgbytes1)
                window['result_cam1'].update(value='')



            if event == 'Webcam2':
                #cap2 = cv2.VideoCapture(1)
                recording2 = True



            elif event == 'Stop2':
                recording2 = False 
                imgbytes2 = np.zeros([100,100,3],dtype=np.uint8)
                imgbytes2 = cv2.resize(imgbytes2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                imgbytes2 = cv2.imencode('.png',imgbytes2)[1].tobytes()
                window['image2'].update(data=imgbytes2)
                window['result_cam2'].update(value='')


            if recording1:
                if values['have_model1'] == True:
                    img1_orgin = my_callback1.image 
                    img1_orgin = img1_orgin[50:530,70:710]
                    img1_orgin = img1_orgin.copy()
                    img1_orgin = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB)                              
                    result1 = model1(img1_orgin,size= values['imgsz1_a22'],conf= values['conf_thres1_a22']/100)             
                    table1 = result1.pandas().xyxy[0]
                    area_remove1 = []

                    for item in range(len(table1.index)):
                        width1 = table1['xmax'][item] - table1['xmin'][item]
                        height1 = table1['ymax'][item] - table1['ymin'][item]
                        area1 = width1*height1
                        if table1['name'][item] == 'nut_me':
                            if area1 < values['area_nutme1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'divat':
                            if area1 < values['area_divat1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'me':
                            if area1 < values['area_me1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'namchamcao':
                            if height1 < values['y_namchamcao1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'tray_bac_truc':
                            if area1 < values['area_traybactruc1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'di_vat_duoi':
                            if area1 < values['area_divatduoi1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'kimcao':
                            if height1 < values['ymin_kimnamcham1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)
                            elif height1 > values['ymax_kimnamcham1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                    
                    names1 = list(table1['name'])
                    print(names1)

                    show1 = np.squeeze(result1.render(area_remove1))
                    show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

                    len_ncc = 0
                    len_kimcao = 0
                    for find_name1 in names1:
                        if find_name1 == 'namchamcao':
                            len_ncc +=1

                        if find_name1 == 'kimcao':
                            len_kimcao += 1


                    if 'kimnamcham' not in names1 or len_ncc !=2 or len_kimcao !=2 \
                    or 'divat' in names1 or 'me' in names1 or 'nut_me' in names1 or 'nut' in names1 \
                    or 'tray_bac_truc' in names1 or 'di_vat_duoi' in names1:
                        print('NG')
                        cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                        window['result_cam1'].update(value= 'NG', text_color='red')

                    else:
                        print('OK')
                        cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam1'].update(value= 'OK', text_color='green')
                    
                    imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                    window['image1'].update(data= imgbytes1)
                else:
                    img1_orgin = my_callback1.image
                    img1_orgin = img1_orgin[50:530,70:710]
                    img1_orgin = img1_orgin.copy()
                    img1_orgin = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB) 
                    img1_resize = cv2.resize(img1_orgin,(image_width_display,image_height_display))
                    if img1_orgin is not None:
                        show1 = img1_resize
                        imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                        window['image1'].update(data=imgbytes1)
                        window['result_cam1'].update(value='')


            if recording2:
                if values['have_model2'] == True:
                    img2_orgin = my_callback2.image
                    
                    result2 = model2(img2_orgin,size= values['imgsz2_a22'],conf= values['conf_thres2_a22']/100) 

                    table2 = result2.pandas().xyxy[0]

                    area_remove2 = []
                    for item in range(len(table2.index)):
                        width2 = table2['xmax'][item] - table2['xmin'][item]
                        height2 = table2['ymax'][item] - table2['ymin'][item]
                        area2 = width2*height2
                        if table2['name'][item] == 'nut_me':
                            if area2 < values['area_nutme2_a22']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                        elif table2['name'][item] == 'divat':
                            if area2 < values['area_divat2_a22']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                        elif table2['name'][item] == 'me':
                            if area2 < values['area_me2_a22']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                    names2 = list(table2['name'])
                    print(names2)

                    show2 = np.squeeze(result2.render(area_remove2))
                    show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

                    if 'namcham' not in names2 \
                    or 'divat' in names2 or 'me' in names2 or 'namchamcao' in names2 or 'nut_me' in names2 or 'nut' in names2:
                        print('NG')
                        cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                        window['result_cam2'].update(value= 'NG', text_color='red')  

                    else:
                        print('OK')
                        cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam2'].update(value= 'OK', text_color='green')


                    imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                    window['image2'].update(data= imgbytes2)
                else:
                    img2_orgin = my_callback2.image
                    img2_resize = cv2.resize(img2_orgin,(image_width_display,image_height_display))
                    if img2_orgin is not None:
                        show2 = img2_resize
                        imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                        window['image2'].update(data=imgbytes2)
                        window['result_cam2'].update(value='')




            if event == 'Pic1':
                dir_img1 = sg.popup_get_file('Choose your image 1',file_types=file_name_img,keep_on_top= True)
                if dir_img1 not in ('',None):
                    pic1 = Image.open(dir_img1)
                    img1_resize = pic1.resize((image_width_display,image_height_display))
                    imgbytes1 = ImageTk.PhotoImage(img1_resize)
                    window['image1'].update(data= imgbytes1)
                    window['Detect1'].update(disabled= False)         

            if event == 'Pic2':
                dir_img2 = sg.popup_get_file('Choose your image 2',file_types=file_name_img,keep_on_top= True)
                if dir_img2 not in ('',None):
                    pic2 = Image.open(dir_img2)
                    img2_resize = pic2.resize((image_width_display,image_height_display))
                    imgbytes2 = ImageTk.PhotoImage(img2_resize)
                    window['image2'].update(data=imgbytes2)
                    window['Detect2'].update(disabled= False)


            if event == 'Change1':
                model1= torch.hub.load('./levu','custom',path=values['file_weights1_a22'],source='local',force_reload=False)
                window['result_cam1'].update(value='A22')

            if event == 'Change2':
                model2= torch.hub.load('./levu','custom',path=values['file_weights2_a22'],source='local',force_reload=False)
                window['result_cam2'].update(value='A22')

            if event == 'Detect1':
                print('CAM 1 DETECT')
                t1 = time.time()
                try:
                    result1 = model1(pic1,size= values['imgsz1_a22'],conf = values['conf_thres1_a22']/100)
                    t2 = time.time() - t1
                    print(t2) 

                    table1 = result1.pandas().xyxy[0]

                    area_remove1 = []
                    for item in range(len(table1.index)):
                        width1 = table1['xmax'][item] - table1['xmin'][item]
                        height1 = table1['ymax'][item] - table1['ymin'][item]
                        area1 = width1*height1

                        if table1['name'][item] == 'nut_me':
                            if area1 < values['area_nutme1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'divat': 
                            if area1 < values['area_divat1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'me':
                            if area1 < values['area_me1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'namchamcao':
                            if height1 < values['y_namchamcao1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'tray_bac_truc':
                            if area1 < values['area_traybactruc1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'di_vat_duoi':
                            if area1 < values['area_divatduoi1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'kimcao':
                            if height1 < values['ymin_kimnamcham1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)
                            elif height1 > values['ymax_kimnamcham1_a22']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                    names1 = list(table1['name'])

                    show1 = np.squeeze(result1.render(area_remove1))
                    show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

                    len_ncc = 0
                    for ncc in names1:
                        if ncc == 'namchamcao':
                            len_ncc +=1
                    
                    len_kimcao = 0
                    for kimcao in names1:
                        if kimcao == 'kimcao':
                            len_kimcao += 1

                    if 'kimnamcham' not in names1 or len_ncc !=2 or len_kimcao !=2 \
                    or 'divat' in names1 or 'me' in names1 or 'nut_me' in names1 or 'nut' in names1 \
                    or 'tray_bac_truc' in names1 or 'di_vat_duoi' in names1:
                        print('NG')
                        cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                        window['result_cam1'].update(value= 'NG', text_color='red')
                    else:
                        print('OK')
                        cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam1'].update(value= 'OK', text_color='green')

                    
                    imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                    window['image1'].update(data= imgbytes1)
                
                except:
                    sg.popup_annoying("Don't have image", font=('Helvetica',14),text_color='red')
                
                t2 = time.time() - t1
                print(t2)
                print('---------------------------------------------') 


                
            if event == 'Detect2':
                print('CAM 2 DETECT')
                t1 = time.time()
                try:
                    result2 = model2(dir_img2,size= values['imgsz2_a22'],conf = values['conf_thres2_a22']/100)
                    t2 = time.time() - t1
                    print(t2) 
                
                    table2 = result2.pandas().xyxy[0]

                    area_remove2 = []
                    for item in range(len(table2.index)):
                        width2 = table2['xmax'][item] - table2['xmin'][item]
                        height2 = table2['ymax'][item] - table2['ymin'][item]
                        area2 = width2*height2
                        if table2['name'][item] == 'nut_me':
                            if area2 < values['area_nutme2_a22']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)
                        
                        elif table2['name'][item] == 'divat':
                            if area2 < values['area_divat2_a22']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                        elif table2['name'][item] == 'me':
                            if area2 < values['area_me2_a22']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                    names2 = list(table2['name'])

                    show2 = np.squeeze(result2.render(area_remove2))
                    show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

                    if 'namcham' not in names2 \
                    or 'divat' in names2 or 'me' in names2 or 'namchamcao' in names2 or 'nut_me' in names2 or 'nut' in names2:
                        print('NG')
                        cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                        window['result_cam2'].update(value= 'NG', text_color='red')       

                    else:
                        print('OK')
                        cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam2'].update(value= 'OK', text_color='green')

                    imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                    window['image2'].update(data= imgbytes2)
                except:
                    sg.popup_annoying("Don't have image", font=('Helvetica',14),text_color='red')

                t2 = time.time() - t1
                print(t2) 
                print('---------------------------------------------')
            



        if values['change_model'] == 'A19':

            if event =='Exit' or event == sg.WINDOW_CLOSED :
                break

            if event == 'Configure':
                if window.TKroot.state() == 'zoomed':
                    #print(window['image1'].get_size()[0])
                    image_width_display = window['image1'].get_size()[0]
                    image_height_display = window['image1'].get_size()[1]
                    result_width_display = image_width_display - 190
                    result_height_display = 100 


            if event =='Administrator':
                login_password = 'vu123'  # helloworld
                password = sg.popup_get_text(
                    'Enter Password: ', password_char='*') 
                if password == login_password:
                    sg.popup_ok('Login Successed!!! ',text_color='green', font=('Helvetica',14))  

                    window['imgsz2_a22'].update(disabled= False)
                    window['imgsz1_a22'].update(disabled= False)
                    window['conf_thres2_a22'].update(disabled= False)
                    window['conf_thres1_a22'].update(disabled= False)
                    window['area_nutme2_a22'].update(disabled= False)
                    window['area_nutme1_a22'].update(disabled= False)
                    window['area_divat2_a22'].update(disabled= False)
                    window['area_divat1_a22'].update(disabled= False)
                    window['area_me2_a22'].update(disabled= False)
                    window['area_me1_a22'].update(disabled= False)
                    window['y_namchamcao1_a22'].update(disabled= False)
                    window['area_traybactruc1_a22'].update(disabled= False)
                    window['area_divatduoi1_a22'].update(disabled= False)
                    window['ymin_kimnamcham1_a22'].update(disabled= False)
                    window['ymax_kimnamcham1_a22'].update(disabled= False)

                    window['imgsz2_a19'].update(disabled= False)
                    window['imgsz1_a19'].update(disabled= False)
                    window['conf_thres2_a19'].update(disabled= False)
                    window['conf_thres1_a19'].update(disabled= False)
                    window['area_nutme2_a19'].update(disabled= False)
                    window['area_nutme1_a19'].update(disabled= False)
                    window['area_divat2_a19'].update(disabled= False)
                    window['area_divat1_a19'].update(disabled= False)
                    window['area_me2_a19'].update(disabled= False)
                    window['area_me1_a19'].update(disabled= False)
                    window['y_namchamcao1_a19'].update(disabled= False)
                    window['area_traybactruc1_a19'].update(disabled= False)
                    window['area_divatduoi1_a19'].update(disabled= False)
                    window['ymin_kimnamcham1_a19'].update(disabled= False)
                    window['ymax_kimnamcham1_a19'].update(disabled= False)

                    window['file_browse2_a22'].update(disabled= False,button_color='turquoise')
                    window['file_browse1_a22'].update(disabled= False,button_color='turquoise')

                    window['file_browse2_a19'].update(disabled= False,button_color='turquoise')
                    window['file_browse1_a19'].update(disabled= False,button_color='turquoise')

                    window['SaveData1_a22'].update(disabled= False,button_color='turquoise')
                    window['SaveData2_a22'].update(disabled= False,button_color='turquoise')

                    window['SaveData1_a19'].update(disabled= False,button_color='turquoise')
                    window['SaveData2_a19'].update(disabled= False,button_color='turquoise')

                    window['Webcam1'].update(disabled= False,button_color='turquoise')
                    window['Webcam2'].update(disabled= False,button_color='turquoise')
                    window['Stop1'].update(disabled= False,button_color='turquoise')
                    window['Stop2'].update(disabled= False,button_color='turquoise')
                    window['Pic1'].update(disabled= False,button_color='turquoise')
                    window['Pic2'].update(disabled= False,button_color='turquoise')
                    window['Snap1'].update(disabled= False,button_color='turquoise')
                    window['Snap2'].update(disabled= False,button_color='turquoise')
                    window['Change1'].update(button_color='turquoise')
                    window['Change2'].update(button_color='turquoise')
                    window['Detect1'].update(button_color='turquoise')
                    window['Detect2'].update(button_color='turquoise')

                    window['have_model1'].update(disabled=False)

                    window['have_model2'].update(disabled=False)

                    window['red_input1'].update(disabled=False)
                    window['red_slider1'].update(disabled=False)        
                    window['choose_red1'].update(disabled=False,button_color='turquoise')



        
                else:
                    sg.popup_cancel('Wrong Password!!!',text_color='red', font=('Helvetica',14))

            if event == 'Change Theme':
                layout_theme = [
                    [sg.Listbox(values= sg.theme_list(), size = (30,20),auto_size_text=18,default_values='Dark',key='theme', enable_events=True)],
                    [
                        [sg.Button('Apply'),
                        sg.Button('Cancel')]
                    ]
                ] 
                window_theme = sg.Window('Change Theme', layout_theme, location=(50,50),keep_on_top=True).Finalize()
                window_theme.set_min_size((300,400))   

                while True:
                    event_theme, values_theme = window_theme.read(timeout=20)
                    if event_theme == sg.WIN_CLOSEG:
                        break

                    if event_theme == 'Apply':
                        theme_choose = values_theme['theme'][0]
                        if theme_choose == 'Default':
                            continue
                        window.close()
                        window = make_window(theme_choose)
                        save_theme(theme_choose)
                        #print(theme_choose)
                    if event_theme == 'Cancel':
                        answer = sg.popup_yes_no('Do you want to exit?')
                        if answer == 'Yes':
                            break
                        if answer == 'No':
                            continue
                window_theme.close()


            if event == 'file_browse1_a22': 
                window['file_weights1_a22'].update(value=values['file_browse1_a22'])
                if values['file_browse1_a22']:
                    window['Change1'].update(disabled=False)
            if event == 'file_browse1_a19': 
                window['file_weights1_a19'].update(value=values['file_browse1_a19'])
                if values['file_browse1_a19']:
                    window['Change1'].update(disabled=False)

            if event == 'file_browse2_a22':
                window['file_weights2_a22'].update(value=values['file_browse2_a22'])
                if values['file_browse2_a22']:
                    window['Change2'].update(disabled=False)

            if event == 'file_browse2_a19':
                window['file_weights2_a19'].update(value=values['file_browse2_a19'])
                if values['file_browse2_a19']:
                    window['Change2'].update(disabled=False)

    #change model 
            if event == 'change_model':
                if values['change_model'] == 'A22':
                    model1 = torch.hub.load('./levu','custom', path= list_param_model1_a22[0], source='local',force_reload =False)
                    model2 = torch.hub.load('./levu','custom', path= list_param_model2_a22[0], source='local',force_reload =False)
                    save_choose_model(values['change_model'])
                    window['result_cam1'].update(value='A22')
                    window['result_cam2'].update(value='A22')

                if values['change_model'] == 'A19':
                    model1 = torch.hub.load('./levu','custom', path= list_param_model1_a19[0], source='local',force_reload =False)
                    model2 = torch.hub.load('./levu','custom', path= list_param_model2_a19[0], source='local',force_reload =False)
                    save_choose_model(values['change_model'])
                    window['result_cam1'].update(value='A19')
                    window['result_cam2'].update(value='A19')


            if event == 'SaveData1_a22':
                save_param_model1_A22(values['file_weights1_a22'], values['imgsz1_a22'],values['conf_thres1_a22'],values['area_nutme1_a22'], values['area_divat1_a22'],values['area_me1_a22'],values['y_namchamcao1_a22'],values['area_traybactruc1_a22'],values['area_divatduoi1_a22'] ,values['ymin_kimnamcham1_a22'],values['ymax_kimnamcham1_a22'])
                sg.popup('Saved param model 1 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


            if event == 'SaveData2_a22':
                save_param_model2_A22(values['file_weights2_a22'],values['imgsz2_a22'],values['conf_thres2_a22'] ,values['area_nutme2_a22'], values['area_divat2_a22'],values['area_me2_a22'])
                sg.popup('Saved param model 2 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


            if event == 'SaveData1_a19':
                save_param_model1_A19(values['file_weights1_a19'], values['imgsz1_a19'],values['conf_thres1_a19'],values['area_nutme1_a19'], values['area_divat1_a19'],values['area_me1_a19'],values['y_namchamcao1_a19'],values['area_traybactruc1_a19'],values['area_divatduoi1_a19'] ,values['ymin_kimnamcham1_a19'],values['ymax_kimnamcham1_a19'])
                sg.popup('Saved param model 1 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)


            if event == 'SaveData2_a19':
                save_param_model2_A19(values['file_weights2_a19'], values['imgsz2_a19'],values['conf_thres2_a19'] ,values['area_nutme2_a19'], values['area_divat2_a19'],values['area_me2_a19'])
                sg.popup('Saved param model 2 successed',font=('Helvetica',15), text_color='green',keep_on_top= True)






            task_camera1_snap(model=model1,size= values['imgsz1_a19'],conf= values['conf_thres1_a19']/100)
            task_camera2_snap(model=model2,size= values['imgsz2_a19'],conf= values['conf_thres2_a19']/100)


            #task1(model1,size= values['imgsz1_a22'],conf= values['conf_thres1_a22']/100)
            #task2(model2,size= values['imgsz2_a22'],conf= values['conf_thres2_a22']/100) 

            # menu

            if event == 'Webcam1':
                #cap1 = cv2.VideoCapture(0)
                recording1 = True


            elif event == 'Stop1':
                recording1 = False 
                imgbytes1 = np.zeros([100,100,3],dtype=np.uint8)
                imgbytes1 = cv2.resize(imgbytes1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                imgbytes1 = cv2.imencode('.png',imgbytes1)[1].tobytes()
                window['image1'].update(data=imgbytes1)
                window['result_cam1'].update(value='')



            if event == 'Webcam2':
                #cap2 = cv2.VideoCapture(1)
                recording2 = True



            elif event == 'Stop2':
                recording2 = False 
                imgbytes2 = np.zeros([100,100,3],dtype=np.uint8)
                imgbytes2 = cv2.resize(imgbytes2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)
                imgbytes2 = cv2.imencode('.png',imgbytes2)[1].tobytes()
                window['image2'].update(data=imgbytes2)
                window['result_cam2'].update(value='')


            if recording1:
                if values['have_model1'] == True:
                    img1_orgin = my_callback1.image 
                    img1_orgin = img1_orgin[50:530,70:710]
                    img1_orgin = img1_orgin.copy()
                    img1_orgin = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB)                              
                    result1 = model1(img1_orgin,size= values['imgsz1_a19'],conf= values['conf_thres1_a19']/100)             
                    table1 = result1.pandas().xyxy[0]
                    area_remove1 = []

                    for item in range(len(table1.index)):
                        width1 = table1['xmax'][item] - table1['xmin'][item]
                        height1 = table1['ymax'][item] - table1['ymin'][item]
                        area1 = width1*height1
                        if table1['name'][item] == 'nut_me':
                            if area1 < values['area_nutme1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'divat':
                            if area1 < values['area_divat1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'me':
                            if area1 < values['area_me1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'namchamcao':
                            if height1 < values['y_namchamcao1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'tray_bac_truc':
                            if area1 < values['area_traybactruc1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'di_vat_duoi':
                            if area1 < values['area_divatduoi1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'kimcao':
                            if height1 < values['ymin_kimnamcham1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)
                            elif height1 > values['ymax_kimnamcham1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                    
                    names1 = list(table1['name'])
                    print(names1)

                    show1 = np.squeeze(result1.render(area_remove1))
                    show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

                    len_ncc = 0
                    len_kimcao = 0
                    for find_name1 in names1:
                        if find_name1 == 'namchamcao':
                            len_ncc +=1

                        if find_name1 == 'kimcao':
                            len_kimcao += 1


                    if 'kimnamcham' not in names1 or len_ncc !=2 or len_kimcao !=2 \
                    or 'divat' in names1 or 'me' in names1 or 'nut_me' in names1 or 'nut' in names1 \
                    or 'tray_bac_truc' in names1 or 'di_vat_duoi' in names1:
                        print('NG')
                        cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                        window['result_cam1'].update(value= 'NG', text_color='red')

                    else:
                        print('OK')
                        cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam1'].update(value= 'OK', text_color='green')
                    
                    imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                    window['image1'].update(data= imgbytes1)
                else:
                    img1_orgin = my_callback1.image
                    img1_orgin = img1_orgin[50:530,70:710]
                    img1_orgin = img1_orgin.copy()
                    img1_orgin = cv2.cvtColor(img1_orgin, cv2.COLOR_BGR2RGB) 
                    img1_resize = cv2.resize(img1_orgin,(image_width_display,image_height_display))
                    if img1_orgin is not None:
                        show1 = img1_resize
                        imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                        window['image1'].update(data=imgbytes1)
                        window['result_cam1'].update(value='')


            if recording2:
                if values['have_model2'] == True:
                    img2_orgin = my_callback2.image
                    
                    result2 = model2(img2_orgin,size= values['imgsz2_a19'],conf= values['conf_thres2_a19']/100) 

                    table2 = result2.pandas().xyxy[0]

                    area_remove2 = []
                    for item in range(len(table2.index)):
                        width2 = table2['xmax'][item] - table2['xmin'][item]
                        height2 = table2['ymax'][item] - table2['ymin'][item]
                        area2 = width2*height2
                        if table2['name'][item] == 'nut_me':
                            if area2 < values['area_nutme2_a19']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                        elif table2['name'][item] == 'divat':
                            if area2 < values['area_divat2_a19']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                        elif table2['name'][item] == 'me':
                            if area2 < values['area_me2_a19']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                    names2 = list(table2['name'])
                    print(names2)

                    show2 = np.squeeze(result2.render(area_remove2))
                    show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

                    if 'namcham' not in names2 \
                    or 'divat' in names2 or 'me' in names2 or 'namchamcao' in names2 or 'nut_me' in names2 or 'nut' in names2:
                        print('NG')
                        cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                        window['result_cam2'].update(value= 'NG', text_color='red')  

                    else:
                        print('OK')
                        cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam2'].update(value= 'OK', text_color='green')


                    imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                    window['image2'].update(data= imgbytes2)
                else:
                    img2_orgin = my_callback2.image
                    img2_resize = cv2.resize(img2_orgin,(image_width_display,image_height_display))
                    if img2_orgin is not None:
                        show2 = img2_resize
                        imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                        window['image2'].update(data=imgbytes2)
                        window['result_cam2'].update(value='')




            if event == 'Pic1':
                dir_img1 = sg.popup_get_file('Choose your image 1',file_types=file_name_img,keep_on_top= True)
                if dir_img1 not in ('',None):
                    pic1 = Image.open(dir_img1)
                    img1_resize = pic1.resize((image_width_display,image_height_display))
                    imgbytes1 = ImageTk.PhotoImage(img1_resize)
                    window['image1'].update(data= imgbytes1)
                    window['Detect1'].update(disabled= False)         

            if event == 'Pic2':
                dir_img2 = sg.popup_get_file('Choose your image 2',file_types=file_name_img,keep_on_top= True)
                if dir_img2 not in ('',None):
                    pic2 = Image.open(dir_img2)
                    img2_resize = pic2.resize((image_width_display,image_height_display))
                    imgbytes2 = ImageTk.PhotoImage(img2_resize)
                    window['image2'].update(data=imgbytes2)
                    window['Detect2'].update(disabled= False)


            if event == 'Change1':
                model1= torch.hub.load('./levu','custom',path=values['file_weights1_a19'],source='local',force_reload=False)
                window['result_cam1'].update(value='A19') 
                #window['Detect1'].update(disabled= False)

            if event == 'Change2':
                model2 = torch.hub.load('./levu','custom',path=values['file_weights2_a19'],source='local',force_reload=False)
                window['result_cam2'].update(value='A19')


            if event == 'Detect1':
                print('CAM 1 DETECT')
                t1 = time.time()
                try:
                    result1 = model1(pic1,size= values['imgsz1_a19'],conf = values['conf_thres1_a19']/100)
                    t2 = time.time() - t1
                    print(t2) 

                    table1 = result1.pandas().xyxy[0]

                    area_remove1 = []
                    for item in range(len(table1.index)):
                        width1 = table1['xmax'][item] - table1['xmin'][item]
                        height1 = table1['ymax'][item] - table1['ymin'][item]
                        area1 = width1*height1

                        if table1['name'][item] == 'nut_me':
                            if area1 < values['area_nutme1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'divat': 
                            if area1 < values['area_divat1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'me':
                            if area1 < values['area_me1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'namchamcao':
                            if height1 < values['y_namchamcao1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'tray_bac_truc':
                            if area1 < values['area_traybactruc1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'di_vat_duoi':
                            if area1 < values['area_divatduoi1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                        elif table1['name'][item] == 'kimcao':
                            if height1 < values['ymin_kimnamcham1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)
                            elif height1 > values['ymax_kimnamcham1_a19']: 
                                table1.drop(item, axis=0, inplace=True)
                                area_remove1.append(item)

                    names1 = list(table1['name'])

                    show1 = np.squeeze(result1.render(area_remove1))
                    show1 = cv2.resize(show1, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

                    len_ncc = 0
                    for ncc in names1:
                        if ncc == 'namchamcao':
                            len_ncc +=1
                    
                    len_kimcao = 0
                    for kimcao in names1:
                        if kimcao == 'kimcao':
                            len_kimcao += 1

                    if 'kimnamcham' not in names1 or len_ncc !=2 or len_kimcao !=2 \
                    or 'divat' in names1 or 'me' in names1 or 'nut_me' in names1 or 'nut' in names1 \
                    or 'tray_bac_truc' in names1 or 'di_vat_duoi' in names1:
                        print('NG')
                        cv2.putText(show1, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                        window['result_cam1'].update(value= 'NG', text_color='red')
                    else:
                        print('OK')
                        cv2.putText(show1, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam1'].update(value= 'OK', text_color='green')

                    
                    imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                    window['image1'].update(data= imgbytes1)
                
                except:
                    sg.popup_annoying("Don't have image", font=('Helvetica',14),text_color='red')
                
                t2 = time.time() - t1
                print(t2)
                print('---------------------------------------------') 


                
            if event == 'Detect2':
                print('CAM 2 DETECT')
                t1 = time.time()
                try:
                    result2 = model2(dir_img2,size= values['imgsz2_a19'],conf = values['conf_thres2_a19']/100)
                    t2 = time.time() - t1
                    print(t2) 
                
                    table2 = result2.pandas().xyxy[0]

                    area_remove2 = []
                    for item in range(len(table2.index)):
                        width2 = table2['xmax'][item] - table2['xmin'][item]
                        height2 = table2['ymax'][item] - table2['ymin'][item]
                        area2 = width2*height2
                        if table2['name'][item] == 'nut_me':
                            if area2 < values['area_nutme2_a19']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)
                        
                        elif table2['name'][item] == 'divat':
                            if area2 < values['area_divat2_a19']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                        elif table2['name'][item] == 'me':
                            if area2 < values['area_me2_a19']: 
                                table2.drop(item, axis=0, inplace=True)
                                area_remove2.append(item)

                    names2 = list(table2['name'])

                    show2 = np.squeeze(result2.render(area_remove2))
                    show2 = cv2.resize(show2, (image_width_display,image_height_display), interpolation = cv2.INTER_AREA)

                    if 'namcham' not in names2 \
                    or 'divat' in names2 or 'me' in names2 or 'namchamcao' in names2 or 'nut_me' in names2 or 'nut' in names2:
                        print('NG')
                        cv2.putText(show2, 'NG',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                        window['result_cam2'].update(value= 'NG', text_color='red')       

                    else:
                        print('OK')
                        cv2.putText(show2, 'OK',(result_width_display,result_height_display),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        window['result_cam2'].update(value= 'OK', text_color='green')

                    imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                    window['image2'].update(data= imgbytes2)
                except:
                    sg.popup_annoying("Don't have image", font=('Helvetica',14),text_color='red')

                t2 = time.time() - t1
                print(t2) 
                print('---------------------------------------------')
            



    try:
        st_device1.acquisition_stop()
        st_datastream1.stop_acquisition()
    except:
        print('Cam 1 is not defined')

    try:
        st_device2.acquisition_stop()
        st_datastream2.stop_acquisition()
    except:
        print('Cam 2 is not defined')

    window.close() 

except Exception as e:
    traceback.print_exc()
    str_error = str(e)
    sg.popup(str_error,font=('Helvetica',15), text_color='red',keep_on_top= True)
                     