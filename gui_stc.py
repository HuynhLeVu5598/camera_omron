from copyreg import remove_extension
import glob
import os
import cv2
from os import path
import threading
import time
import socket
from time import sleep
import torch
import numpy as np 
import pathlib
import sys

import PySimpleGUI as sg
from PySimpleGUI.PySimpleGUI import WIN_CLOSED, Checkbox
from PIL import Image,ImageTk
import io 

#import stapipy as st

# Image scale when displaying using OpenCV.
DISPLAY_RESIZE_FACTOR = 0.3

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

    def datastream_callback(self, handle=None, context=None):
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
                                       fx=DISPLAY_RESIZE_FACTOR,
                                       fy=DISPLAY_RESIZE_FACTOR)
                    self._lock.acquire()
                    self._image = nparr
                    self._lock.release()


def setup_camera_stc():
    try:
        st.initialize()
        st_system = st.create_system()
        st_device = st_system.create_first_device()
        print('Device=', st_device.info.display_name)
        st_datastream = st_device.create_datastream()
        callback = st_datastream.register_callback(cb_func)
        st_datastream.start_acquisition()
        st_device.acquisition_start()
        remote_nodemap = st_device.remote_port.nodemap
        return  st_datastream, st_device,remote_nodemap

    except Exception as exception:
        print(exception)
        #return  st_datastream, st_device,remote_nodemap


soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#soc.settimeout(5)

def socket_connect(host, port):
    try:
        soc.connect((host, port))
        return True
    except OSError:
        print("Can't connect to PLC")
        sleep(3)
        print("Reconnecting....")
        return False

def readdata(data):
    a = 'RD '
    c = '\x0D'
    d = a+ data +c
    datasend = d.encode("UTF-8")
    soc.sendall(datasend)
    data = soc.recv(1024)
    datadeco = data.decode("UTF-8")
    data1 = int(datadeco)

    return data1

#Write data
def writedata(register, data):
    a = 'WR '
    b = ' '
    c = '\x0D'
    d = a+ register + b + str(data) + c
    datasend  = d.encode("UTF-8")
    soc.sendall(datasend)
    datares = soc.recv(1024)
    #print(datares)

def removefile():
    directory1 = 'D:/FH1/'
    directory2 = 'D:/FH2/'
    if os.listdir(directory1) != []:
        for i in glob.glob(directory1+'*'):
            os.remove(i)

    if os.listdir(directory2) != []:
        for i in glob.glob(directory2+'*'):
            os.remove(i)

    directory3 = 'D:/FH/camera1/'
    directory4 = 'D:/FH/camera2/'
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

def take_name_image(directory):
    #directory5 = 'D:/nc/sava_data/'
    if os.listdir(directory) == []:
        i =0
    else:
        path = pathlib.Path(directory)
        # path.stat().st_mtime
        time, file_path = max((f.stat().st_mtime,f) for f in path.iterdir())
        i = int(file_path.stem)
    return i


def task1():
    if readdata('DM1400')==1000: 
        print('1')
        for filename1 in glob.glob('D:/FH/camera1/*'):
            for path1 in glob.glob(filename1 + '/*.jpg'):
                img1 = cv2.imread(path1)
                img1 = cv2.resize(img1,(640,480))   
                cv2.imshow('image1',img1)
                cv2.waitKey(1)    
                os.remove(path1)
            os.rmdir(filename1)
        writedata('DM1500.U',2000)  
        writedata('DM1400.U',2000) 


def task2():
    if readdata('DM2400')==1000:
        print('2')
        for filename2 in glob.glob('D:/FH/camera2/*'):
            for path2 in glob.glob(filename2 + '/*.jpg'):
                img2 = cv2.imread(path2)
                img2 = cv2.resize(img2,(640,480))
                cv2.imshow('image2',img2)
                cv2.waitKey(1)    
                os.remove(path2)
            os.rmdir(filename2)
        writedata('DM2500.U',2000)  
        writedata('DM2400.U',2000) 


# have PLC
def task3(model, size, conf, i):
    if readdata('DM1400')==1000:
        directory3 = 'D:/FH/camera1/'
        if os.listdir(directory3) == []:
            print('folder 1 empty')
        else:
            #a = time.time()
            for filename1 in glob.glob('D:/FH/camera1/*'):
                for path1 in glob.glob(filename1 + '/*.jpg'):
                    img1 = cv2.imread(path1)
                    while type(img1) == type(None):
                        print('error 1')
                        #for filename1 in glob.glob('D:/FH/camera1/*'):
                        for path1 in glob.glob(filename1 + '/*.jpg'):
                            img1 = cv2.imread(path1)
                    cv2.imwrite('D:/nc/save_data5/' + str(i) + '.jpg',img1)
                    #i+=1
                    img1 = cv2.resize(img1,(640,480))
                    result1 = model(img1,size= size,conf = conf) 
                    for i0, pred in enumerate(result1.pred):
                        if pred.shape[0]:
                            for *box,cof,clas in reversed(pred):
                                #print(clas)
                                if result1.names[int(clas.tolist())] == 'bactruc':
                                    print('ok1')
                                    #writedata('MR102',1)
                                    #time.sleep(0.3)
                                    writedata('DM1500.U',2000)  
                                    writedata('DM1400.U',2000) 
                                else:
                                    print('ng1')
                                    #writedata('MR103',1)
                                    #time.sleep(0.3)
                                    writedata('DM1500.U',2000)  
                                    writedata('DM1400.U',2000)
                        else:
                            print('ng11')
                            #writedata('MR103',1)
                            #time.sleep(0.3)
                            writedata('DM1500.U',2000)  
                            writedata('DM1400.U',2000) 
                    show1 = np.squeeze(result1.render())
                    cv2.imshow('image1',show1)
                    #b = time.time() - a
                    #print(str(b))
                    cv2.waitKey(1)    
                    os.remove(path1)

                os.rmdir(filename1)


def task4(model, size, conf, j):
    if readdata('DM2400')==1000:
        directory4 = 'D:/FH/camera2/'
        if os.listdir(directory4) == []:
            print('folder 2 empty')
        else:
            for filename2 in glob.glob('D:/FH/camera2/*'):
                for path2 in glob.glob(filename2 + '/*.jpg'):
                    img2 = cv2.imread(path2)
                    while type(img2) == type(None):
                        print('error 2')
                        #for filename2 in glob.glob('D:/FH/camera2/*'):
                        for path2 in glob.glob(filename2 + '/*.jpg'):
                            img2 = cv2.imread(path2)
                    #img2 = cv2.resize(img2,(640,480))
                    #j+=1
                    cv2.imwrite('D:/nc/save_data6/' + str(j) + '.jpg',img2)
                    #j+=1
                    img2 = cv2.resize(img2,(640,480))
                    result2 = model(img2,size= size,conf = conf) 
                    for i0, pred in enumerate(result2.pred):
                        if pred.shape[0]:
                            for *box,cof,clas in reversed(pred):
                                #print(clas)
                                if result2.names[int(clas.tolist())] == 'bactruc':
                                    print('ok2')
                                    #writedata('MR104',1)
                                    #time.sleep(0.3)
                                    writedata('DM2500.U',2000) 
                                    writedata('DM2400.U',2000) 
                                    print('---')
                                else:
                                    print('ng2')
                                    #writedata('MR105',1)
                                    #time.sleep(0.3)
                                    writedata('DM2500.U',2000)  
                                    writedata('DM2400.U',2000) 
                                    print('---')
                        else:
                            print('ng22')
                            #writedata('MR105',1)
                            #time.sleep(0.3)
                            writedata('DM2500.U',2000)  
                            writedata('DM2400.U',2000) 
                            print('---')
                    show2 = np.squeeze(result2.render())
                    cv2.imshow('image2',show2)
                    cv2.waitKey(1)    
                    os.remove(path2)
                os.rmdir(filename2)




# don't PLC
def task5(model, size, conf,max_det, classes):
    for filename1 in glob.glob('D:/FH/camera1/*'):
        for path1 in glob.glob(filename1 + '/*.jpg'):
            #img1 = Image.open(path1)
            #img1.thumbnail((640,480))
            #img1 = ImageTk.PhotoImage(img1)
            img1 = cv2.imread(path1)
            img1 = cv2.resize(img1,(640,480))
            result1 = model(img1,size= size,conf = conf, max_det= max_det, classes= classes)
            # for i0, pred in enumerate(result1.pred):
            #     if pred.shape[0]:
            #         for *box,cof,clas in reversed(pred):
            #             if result1.names[int(clas.tolist())] == 'person':
            #                 print('ok1')
            #             else:
            #                 print('ng1')
            #     else:
            #         print('ng11')
            for i, pred in enumerate(result1.pred):
                if pred.shape[0]:
                    window['result1'].update('NG')
                else:
                    window['result1'].update('OK')

            show1 = np.squeeze(result1.render())
            #cv2.imshow('image1',show1)
            #cv2.waitKey(1)
            imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
            window['image1'].update(data=imgbytes1)
            os.remove(path1)
        os.rmdir(filename1)



def task6(model, size, conf, max_det, classes):
    for filename2 in glob.glob('D:/FH/camera2/*'):
        for path2 in glob.glob(filename2 + '/*.jpg'):
            img2 = cv2.imread(path2)
            img2 = cv2.resize(img2,(640,480))
            result2 = model(img2,size= size,conf = conf, max_det= max_det, classes= classes)
            # for i0, pred in enumerate(result2.pred):
            #     if pred.shape[0]:
            #         for *box,cof,clas in reversed(pred):
            #             if result2.names[int(clas.tolist())] == 'person':
            #                 print('ok2')
            #             else:
            #                 print('ng2')
            #     else:
            #         print('ng22')

            for i, pred in enumerate(result2.pred):
                if pred.shape[0]:
                    window['result2'].update('NG')
                else:
                    window['result2'].update('OK')
            

            show2 = np.squeeze(result2.render())
            #cv2.imshow('image2',show2)
            #cv2.waitKey(1)    
            imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
            window['image2'].update(data=imgbytes2)
            os.remove(path2)
        os.rmdir(filename2)


# PLC
def PLC1(model,size,conf,max_det,classes):
    if readdata('DM1400')==1000:
        directory3 = 'D:/FH/camera1/'
        if os.listdir(directory3) != []:
            for filename1 in glob.glob('D:/FH/camera1/*'):
                for path1 in glob.glob(filename1 + '/*.jpg'):
                    img1 = cv2.imread(path1)
                    while type(img1) == type(None):
                        for path1 in glob.glob(filename1 + '/*.jpg'):
                            img1 = cv2.imread(path1)
                    img1 = cv2.resize(img1,(640,480))
                    result1 = model(img1,size= size,conf = conf,max_det = max_det,classes= classes) 
                    for i, pred in enumerate(result1.pred):
                        if pred.shape[0]:
                            window['result1'].update('NG')
                            writedata('DM1500.U',2000)  
                            writedata('DM1400.U',2000) 
                        else:
                            window['result1'].update('OK')
                            writedata('DM1500.U',2000)  
                            writedata('DM1400.U',2000) 
                    show1 = np.squeeze(result1.render())
                    imgbytes1 = show1
                    window['image1'].update(data= imgbytes1)
                    os.remove(path1)

                os.rmdir(filename1)


def PLC2(model, size, conf, max_det, classes):
    if readdata('DM2400')==1000:
        directory4 = 'D:/FH/camera2/'
        if os.listdir(directory4) != []:
            for filename2 in glob.glob('D:/FH/camera2/*'):
                for path2 in glob.glob(filename2 + '/*.jpg'):
                    img2 = cv2.imread(path2)
                    while type(img2) == type(None):
                        for path2 in glob.glob(filename2 + '/*.jpg'):
                            img2 = cv2.imread(path2)
                    img2 = cv2.resize(img2,(640,480))
                    result2 = model(img2,size= size,conf = conf,max_det = max_det,classes= classes) 
                    for i, pred in enumerate(result2.pred):
                        if pred.shape[0]:
                            window['result1'].update('NG')
                            writedata('DM2500.U',2000)  
                            writedata('DM2400.U',2000) 
                        else:
                            window['result2'].update('OK')
                            writedata('DM2500.U',2000)  
                            writedata('DM2400.U',2000) 
                    show2 = np.squeeze(result2.render())
                    imgbytes2 = show2
                    window['image2'].update(data= imgbytes2) 
                    os.remove(path2)
                os.rmdir(filename2)


def connect_PLC():
    if have_model1 == True and have_model2 == True:
        connected = False
        while connected == False:
            connected = socket_connect('192.168.0.20',8501)
        print("connected")   
        sg.popup('Connect plc successed', font=('Helvetica',15),text_color= 'lime')

        removefile()
        while True:
            PLC1(model1, values['imgsz1'], values['conf_thres1']/100,values['max_det1'], values_classes1)
            PLC2(model2, values['imgsz2'], values['conf_thres2']/100,values['max_det2'], values_classes2)
            if event == 'Interrupt Connect PLC':
                break
    else:
        sg.popup("Don't have model",font=('Helvetica',15), text_color= 'lime')
        print("Don't have model")
def subcheck():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('check cam',frame)
            if cv2.waitKey(1) &0xff == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return True
                #break

        else:
            return False

def checkcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('check cam',frame)
            if cv2.waitKey(1) &0xff == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                break
        else:
            answer = sg.popup_yes_no('Cam no connect\n Do you want check again?') 
            if answer == 'No':
                #time.sleep(1)
                cap.release()
                cv2.destroyAllWindows()
                break
            elif answer == 'Yes':
                #time.sleep(1)
                result_check = subcheck()
                if result_check == False:
                    continue
                else:
                    break


def config_off_auto(remote_nodemap):
    # Configure the ExposureMode
    node_name1 = "ExposureMode"

    node1 = remote_nodemap.get_node(node_name1)

    if not node1.is_writable:
        print('not node 1')
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
        print('not node 2')
    enum_node2 = st.PyIEnumeration(node2)
    enum_entries2 = enum_node2.entries
    selection2 = 0

    if selection2 < len(enum_entries2):
        enum_entry2 = enum_entries2[selection2]
        enum_node2.set_int_value(enum_entry2.value)

    # Configure the BalanceWhiteAuto
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
        

def BalanceWhiteAuto(name, name1,name_button,index):
    if event_control == name1:
        values_control[name] = values_control[name1]
        window_control[name].update(value = values_control[name1])
    if event_control == name:
        values_control[name1] = values_control[name]
        window_control[name1].update(value = values_control[name])
        
    if event_control == name_button:
        if values_control[name] not in range(0,255):
            values_control[name] = values_control[name1]
            window_control[name].update(value = values_control[name1])
        enum_name1 = "BalanceRatioSelector"
        numeric_name1 = "BalanceRatio"
        node1 = remote_nodemap.get_node(enum_name1)
        if not node1.is_writable:
            print('not node 1 '+ enum_name1)

        enum_node1 = st.PyIEnumeration(node1)
        enum_entries1 = enum_node1.entries

        enum_entry1 = enum_entries1[index]
        if enum_entry1.is_available:
            enum_node1.value = enum_entry1.value
            #print(st.PyIEnumEntry(enum_entry).symbolic_value)
            node_name2 = numeric_name1
            node2 = remote_nodemap.get_node(node_name2)

            if not node2.is_writable:
                print('not node 2 '+ name)
            else:
                if node2.principal_interface_type == st.EGCInterfaceType.IFloat:
                    node_value2 = st.PyIFloat(node2)
                elif node2.principal_interface_type == st.EGCInterfaceType.IInteger:
                    node_value2 = st.PyIInteger(node2)
                value2 = values_control[name]

                if node2.principal_interface_type == st.EGCInterfaceType.IFloat:
                    value2 = float(value2)
                else:
                    value2 = int(value2)
                if node_value2.min <= value2 <= node_value2.max:
                    node_value2.value = value2


def set_exposure_or_gain(node_name,node_name1,value):
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
            print(node_name + ' : ' + str(node_value.value))

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
    node1 = remote_nodemap.get_node(enum_name1)
    if not node1.is_writable:
        print('not node 1 '+ enum_name1)

    enum_node1 = st.PyIEnumeration(node1)
    enum_entries1 = enum_node1.entries

    enum_entry1 = enum_entries1[index]
    if enum_entry1.is_available:
        enum_node1.value = enum_entry1.value

        node_name2 = numeric_name1
        node2 = remote_nodemap.get_node(node_name2)

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

def set_config_init():
    list_values = []
    with open('config.txt') as lines:
        for line in lines:
            _, value = line.strip().split(':')
            list_values.append(value)
    set_exposure_or_gain("ExposureTime","ExposureTimeRaw",list_values[0])
    set_exposure_or_gain("Gain","GainRaw",list_values[1])
    set_balance_white_auto(0,list_values[2])
    set_balance_white_auto(1,list_values[3])
    set_balance_white_auto(2,list_values[4])
    set_balance_white_auto(3,list_values[5])
    set_balance_white_auto(4,list_values[6])

def return_param_init():
    list_values = []
    with open('config.txt') as lines:
        for line in lines:
            _,value = line.strip().split(':')
            list_values.append(value)
    return list_values

def save_config(v1,v2,v3,v4,v5,v6,v7):
    line1 = 'exposure:' + str(v1)
    line2 = 'gain:' + str(v2)
    line3 = 'red:' + str(v3)
    line4 = 'greenr:' + str(v4)
    line5 = 'greenb:' + str(v5)
    line6 = 'green:' + str(v6)
    line7 = 'blue:' + str(v7)
    lines = [line1,line2,line3,line4,line5,line6 ,line7]
    with open("config.txt", "w") as f:
        for i in lines:
            f.write(i)
            f.write('\n')


def make_window(theme):
    sg.theme(theme)

    CLASSES1 = []
    CLASSES2 = []

    list_spin = [i for i in range(101)]
    file_img = [("JPEG (*.jpg)",("*jpg","*.png"))]

    file_weights = [('Weights (*.pt)', ('*.pt'))]

    menu = [['Application', ['Control','Connect PLC','Interrupt Connect PLC','Exit']],
            ['Tool', ['Check Cam','Change Theme']],
            ['Help',['About']]]

    right_click_menu = [[], ['Exit']]

    

    layout0 = [
        [sg.MenubarCustom(menu, font='Helvetica',text_color='white',background_color='#404040',bar_text_color='white',bar_background_color='#404040',bar_font='Helvetica')],
        [sg.Text('Detection', size =(70,1),justification='center' ,font= ('Helvetica',30),text_color='red' ,relief= sg.RELIEF_SUNKEN)],
        [
            sg.Image(filename='', size=(640,480),key='image1',background_color='black'),
            sg.Image(filename='', size=(640,480),key='image2',background_color='black'),
            sg.Frame('',
            [
                #[sg.Text('Handle', size=(15,1),justification='center',font=('Helvetica', 30), text_color='pink')],
                #[sg.Text('')],
                #[sg.Text('')],
                [sg.Button('Webcam1', size=(10,1),  font=('Helvetica',15),disabled=True),
                sg.Button('Webcam2', size=(10,1),  font=('Helvetica',15),disabled=True)],
                [sg.Text('')],
                [sg.Button('Stop1', size=(10,1), font=('Helvetica',15)),
                sg.Button('Stop2', size=(10,1), font=('Helvetica',15))],
                [sg.Text('')],
                [sg.Button('Change1', size=(10,1),  font=('Helvetica',15)),
                sg.Button('Change2', size=(10,1), font = ('Helvetica',15))],
                [sg.Text('')],
                #[sg.Text('')],
                [sg.Button('Auto', size=(10,1), font=('Helvetica',15), disabled=True),
                sg.Button('Interrupt', size=(10,1), font=('Helvetica',15))],
                [sg.Text('')],
                [sg.Button('Snap1', size=(10,1), font=('Helvetica',15)),
                sg.Button('Snap2', size=(10,1), font=('Helvetica',15))],
                [sg.Text('')],
                [sg.Button('Continue1', size=(10,1), font=('Helvetica',15)),
                sg.Button('Continue2', size=(10,1), font=('Helvetica',15))],
                [sg.Text('')],
                [sg.Button('Exit', size=(10,1), font=('Helvetica',15))]
            ]),
        ],

        [
            sg.Frame('Parameter1',
            [
                #[sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color= 'yellow'), sg.InputOptionMenu(('something','bactruc','vonho'),size=(20,20),default_value='something',key='weights')],
                [sg.Text('Weights1', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(19,1), font=('Helvetica',12), key='file_weights1',readonly= True, text_color='navy') ,sg.FileBrowse(file_types= file_weights, size=(10,1), font=('Helvetica',10))],
                [sg.Text('Size1', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(23,20),default_value=416,key='imgsz1')],
                [sg.Text('Confidence1',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(20,20),default_value=25, key= 'conf_thres1')],
                #[sg.Text('IOU',size=(12,1),font=('Helvetica',15),text_color='yellow'), sg.Slider(range=(1,100),orientation='h', size=(20,20),default_value=45, key='iou_thres')],
                [sg.Text('Max detection1', size=(12,1), font=('Helvetica',15), text_color='yellow'), sg.Spin(values=list_spin, initial_value=10, size=(23,20),key='max_det1')],
                [sg.Text('Classes1', size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Listbox(values=CLASSES1,size=(23,4), text_color= 'aqua',select_mode= sg.LISTBOX_SELECT_MODE_MULTIPLE, key='classes1')],
                [sg.Text('Result1',size=(12,1),font=('Helvetica',15),text_color='yellow'),sg.InputText('',size=(16,20),justification='center',font=('Helvetica',15),text_color='red',readonly=True, key='result1')]
            ],font=('Helvetica',20),title_color='orange'),
            sg.Text(' '  * 50),
            sg.Frame('Parameter2',
            [
                #[sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color= 'yellow'), sg.InputOptionMenu(('something','bactruc','vonho'),size=(20,20),default_value='something',key='weights')],
                [sg.Text('Weights2', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(19,1), font=('Helvetica',12), key='file_weights2',readonly= True, text_color='navy') ,sg.FileBrowse(file_types= file_weights, size=(10,1), font=('Helvetica',10))],
                [sg.Text('Size2', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(23,20),default_value=416,key='imgsz2')],
                [sg.Text('Confidence2',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(20,20),default_value=25, key= 'conf_thres2')],
                #[sg.Text('IOU',size=(12,1),font=('Helvetica',15),text_color='yellow'), sg.Slider(range=(1,100),orientation='h', size=(20,20),default_value=45, key='iou_thres')],
                [sg.Text('Max detection2', size=(12,1), font=('Helvetica',15), text_color='yellow'), sg.Spin(values=list_spin, initial_value=10, size=(23,20),key='max_det2')],
                [sg.Text('Classes2', size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Listbox(values=CLASSES2,size=(23,4), text_color= 'aqua',select_mode= sg.LISTBOX_SELECT_MODE_MULTIPLE, key='classes2')],
                [sg.Text('Result2',size=(12,1),font=('Helvetica',15),text_color='yellow'),sg.InputText('',size=(16,20),justification='center',font=('Helvetica',15),text_color='red',readonly=True, key='result2')]
            ],font=('Helvetica',20),title_color='orange'),
            sg.Text(' '*20),
        ]
    ] 

    
    layout_main = [
        [sg.MenubarCustom(menu, font='Helvetica',text_color='white',background_color='#404040',bar_text_color='white',bar_background_color='#404040',bar_font='Helvetica')],
        [sg.Text('Detection', size =(80,1),justification='center' ,font= ('Helvetica',30),text_color='red' ,relief= sg.RELIEF_SUNKEN)],
        [
        #1
        sg.Frame('',[
            [sg.Image(filename='', size=(640,480),key='image1',background_color='black')],
            [sg.Frame('',
            [
                #[sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color= 'yellow'), sg.InputOptionMenu(('something','bactruc','vonho'),size=(20,20),default_value='something',key='weights')],
                #[sg.Text('Weights1', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(19,1), font=('Helvetica',12), key='file_weights1',readonly= True, text_color='navy'),sg.FileBrowse(file_types= file_weights, size=(10,1), font=('Helvetica',10),key= 'file_browse')],
                [sg.Text('Weights1', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(19,1), font=('Helvetica',12), key='file_weights1',readonly= True, text_color='navy',enable_events= True),
                sg.Frame('',[
                    [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse1',enable_events=True)]
                ], relief= sg.RELIEF_FLAT)],
                [sg.Text('Size1', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(20,20),font=('Helvetica',11),default_value=416,key='imgsz1')],
                [sg.Text('Confidence1',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(20,20),font=('Helvetica',11), default_value=25, key= 'conf_thres1')],
                #[sg.Text('IOU',size=(12,1),font=('Helvetica',15),text_color='yellow'), sg.Slider(range=(1,100),orientation='h', size=(20,20),default_value=45, key='iou_thres')],
                [sg.Text('Max detection1', size=(12,1), font=('Helvetica',15), text_color='yellow'), sg.Spin(values=list_spin, initial_value=10, size=(20,20),font=('Helvetica',11),key='max_det1')],
                [sg.Text('Classes1', size=(12,1),font=('Helvetica',15), text_color='yellow',), sg.Listbox(values=CLASSES1,size=(23,3), text_color= 'aqua',select_mode= sg.LISTBOX_SELECT_MODE_MULTIPLE, key='classes1',enable_events=True), 
                    sg.Checkbox('Remove all',size= (9,1), font= ('Helvetica',15),text_color='yellow', key= 'remove_all_listbox')],
                [sg.Text('Result1',size=(12,1),font=('Helvetica',15),text_color='yellow'),sg.InputText('',size=(16,20),justification='center',font=('Helvetica',15),text_color='red',readonly=True, key='result1')]
            #],font=('Helvetica',15),title_color='orange', expand_y= True),
            ]),
            sg.Frame('',
            [
                #[sg.Text('')],
                [sg.Checkbox('Continous1', size=(10,1),font=('Helvetica',15),text_color='yellow',key= 'Continous1',disabled=True)], 
                [sg.Combo(values=(1,10,100,1000), default_value=1, size=(4,1),font=('Helvetica',12),readonly=True,key='time_continous1',enable_events= True), sg.Text('ms', font=('Helvetica',15), text_color='yellow')]
            ],expand_y= True, relief= sg.RELIEF_FLAT,)
            ]
        ], expand_y= True),
        
        #2
        sg.Frame('',[
            #[sg.Text('')],
            [sg.Button('Webcam1', size=(8,1),  font=('Helvetica',14))],
            [sg.Text('')],
            [sg.Button('Stop1', size=(8,1), font=('Helvetica',14))],
            [sg.Text('')],
            [sg.Button('Change1', size=(8,1),  font=('Helvetica',14),disabled=True)],
            [sg.Text('')],
            [sg.Button('Snap1', size=(8,1), font=('Helvetica',14))],
            [sg.Text('')],
            [sg.Button('Continue1', size=(8,1), font=('Helvetica',14),disabled= True)],
            [sg.Text('')],
            [sg.Button('Pic1', size=(8,1), font=('Helvetica',14))],
            [sg.Text('')],
            [sg.Button('Detect1', size=(8,1), font=('Helvetica',14))],
            #[sg.Text('')],
            #[sg.Checkbox('Continuos1',size=(6,1),font=('Helvetica',12))]
            #[sg.Button('Exit', size=(9,1), font=('Helvetica',15))]
        ],expand_y=True,element_justification='center',expand_x=True),
        #3
        sg.Frame('',[
            [sg.Image(filename='', size=(640,480),key='image2',background_color='black')],
            [sg.Frame('Parameter2',
            [
                #[sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color= 'yellow'), sg.InputOptionMenu(('something','bactruc','vonho'),size=(20,20),default_value='something',key='weights')],
                [sg.Text('Weights2', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(19,1), font=('Helvetica',12), key='file_weights2',readonly= True, text_color='navy') ,sg.FileBrowse(file_types= file_weights, size=(10,1), font=('Helvetica',10))],
                [sg.Text('Size2', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(23,20),default_value=416,key='imgsz2')],
                [sg.Text('Confidence2',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(20,20),default_value=25, key= 'conf_thres2')],
                #[sg.Text('IOU',size=(12,1),font=('Helvetica',15),text_color='yellow'), sg.Slider(range=(1,100),orientation='h', size=(20,20),default_value=45, key='iou_thres')],
                [sg.Text('Max detection2', size=(12,1), font=('Helvetica',15), text_color='yellow'), sg.Spin(values=list_spin, initial_value=10, size=(23,20),key='max_det2')],
                [sg.Text('Classes2', size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Listbox(values=CLASSES2,size=(23,3), text_color= 'aqua',select_mode= sg.LISTBOX_SELECT_MODE_MULTIPLE, key='classes2')],
                [sg.Text('Result2',size=(12,1),font=('Helvetica',15),text_color='yellow'),sg.InputText('',size=(16,20),justification='center',font=('Helvetica',15),text_color='red',readonly=True, key='result2')]
            ],font=('Helvetica',15),title_color='orange',expand_y=True)
            ]
        ]),
        #4
        sg.Frame('',[
            [sg.Button('Webcam2', size=(8,1),  font=('Helvetica',14),disabled=True)],
            [sg.Text('')],
            [sg.Button('Stop2', size=(8,1), font=('Helvetica',14))],
            [sg.Text('')],
            [sg.Button('Change2', size=(8,1),  font=('Helvetica',14), disabled=True)],
            [sg.Text('')],
            [sg.Button('Snap2', size=(8,1), font=('Helvetica',14))],
            [sg.Text('')],
            [sg.Button('Continue2', size=(8,1), font=('Helvetica',14))],
            [sg.Text('')],
            [sg.Button('Pic2', size=(8,1), font=('Helvetica',14))],
            [sg.Text('')],
            [sg.Button('Detect2', size=(8,1), font=('Helvetica',14),disabled= True)],

        ],expand_y=True,element_justification='center',expand_x=True)]
    ]

    layout_terminal = [[sg.Text("Anything printed will display here!")],
                      [sg.Multiline( font=('Helvetica',14), expand_x=True, expand_y=True, write_only=True, autoscroll=True, auto_refresh=True,reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True)]
                      # [sg.Output(size=(60,15), font='Courier 8', expand_x=True, expand_y=True)]
                      ]
    
    graph = sg.Graph(canvas_size=(500,500),graph_bottom_left=(0,0), graph_top_right=(500,500),enable_events=True, key= 'graph')

    layout_image = [
        [sg.Text('Display Image Ng', font=('Helvetica',14))],
        [sg.Button('Check',size=(8,1), font=('Helvetica',14))],
        [graph]    
    ]

    layout = [[sg.TabGroup([[  sg.Tab('Main', layout_main),
                               sg.Tab('Image NG', layout_image),
                               sg.Tab('Output', layout_terminal)]], expand_x=True, expand_y=True)
               ]]


    window = sg.Window('HuynhLeVu', layout, location=(0,0),right_click_menu=right_click_menu).Finalize()
    window.Maximize()

    img = Image.open('image.jpg')
    img1_org = img.resize((640,480))
    imgbytes1 = ImageTk.PhotoImage(img1_org)
    imgbytes2 = imgbytes1
    img2_org = img1_org

    window['image1'].update(data=imgbytes1)
    window['image2'].update(data=imgbytes2)
    return window,img1_org , img2_org


file_name_img = [("Img(*.jpg,*.png)",("*jpg","*.png"))]

recording1 = False
recording2 = False 

continous1 = False
continous2 = False

have_model1 = False
have_model2 = False

flag1 = False
flag2 = False

flag_listbox = False

theme = 'Dark'
window,img1_org, img2_org = make_window(theme)

values_classes1=[]
values_classes2=[]

try:
    my_callback = CMyCallback()
    cb_func = my_callback.datastream_callback
    st_datastream, st_device, remote_nodemap = setup_camera_stc()
    config_off_auto(remote_nodemap)
    set_config_init()
except Exception as exception:
    print(exception)

while True:
    event, values = window.read(timeout=20)

    if event =='Exit' or event == sg.WIN_CLOSED:
        break

    # webcam 1

    if event == 'Webcam1':
        #cap1 = cv2.VideoCapture(0)
        recording1 = True
        window['Continue1'].update(disabled=False)

    elif event == 'Stop1':
        recording1 = False 
        img1 = Image.open('image.jpg')
        img1_org = img1.resize((640,480))
        imgbytes1 = ImageTk.PhotoImage(img1_org)
        window['image1'].update(data=imgbytes1)
        window['result1'].update('')

    if event == 'classes1':
        values_classes1=[]
        #print(values['classes1'])
        for i in values['classes1']:
            values_classes1.append(CLASSES1.index(i))

    if event == 'Change1':
        model1= torch.hub.load('./levu','custom',path=values['file_weights1'],source='local',force_reload=False)
        CLASSES1= model1.names
        window['classes1'].update(values=CLASSES1,set_to_index= [i for i in range(len(CLASSES1))])
        values['classes1'] = [i for i in range(len(CLASSES1))]
        values_classes1 = [i for i in range(len(CLASSES1))]
        continous1 = True
        flag_listbox = True
        have_model1 = True

        if CLASSES1 is not None:
            window['Detect1'].update(disabled= False)
            flag1 = True

    if values['remove_all_listbox'] == True:
        if flag_listbox == True:
            window['classes1'].update(set_to_index=[])
            values['classes1'] = []
            values_classes1 = []

    if recording1:
        if have_model1 == True:
            img1_org = my_callback.image
            img1_org = cv2.resize(img1_org,(640,480))
            if img1_org is not None:
                result1 = model1(img1_org,size= values['imgsz1'],conf = values['conf_thres1']/100, max_det= values['max_det1'], classes= values_classes1)
                for i, pred in enumerate(result1.pred):
                    if pred.shape[0]:
                        window['result1'].update('NG')
                    else:
                        window['result1'].update('OK')

                show1 = np.squeeze(result1.render()) 
                imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                window['image1'].update(data=imgbytes1)
        else:
            img1_org = my_callback.image
            img1_org = cv2.resize(img1_org,(640,480))

            if img1_org is not None:
                show1 = img1_org
                imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                window['image1'].update(data=imgbytes1)
            
          
    if event == 'Snap1':
        #img1_org = frame1
        #img.thumbnail((640,480))
        #img = ImageTk.PhotoImage(img)
        if recording1 == True:
            imgbytes1 = cv2.imencode('.png',img1_org)[1].tobytes()
            window['image1'].update(data = imgbytes1)
            recording1 = False

    if event == 'Continue1':
        recording1 = True

    if event == 'Pic1':
        dir_img1 = sg.popup_get_file('Choose your image 1',file_types=file_name_img,keep_on_top= True)
        if dir_img1 not in ('',None):
            pic1 = Image.open(dir_img1)
            img1_org = pic1.resize((640,480))
            imgbytes1 = ImageTk.PhotoImage(img1_org)
            window['image1'].update(data= imgbytes1)
            window['Detect1'].update(disabled= False)
            #print(dir_img1)

    
    if event == 'Detect1':
        if have_model1 == True:
            model1= torch.hub.load('./levu','custom',path=values['file_weights1'],source='local',force_reload=False)
            result1 = model1(img1_org,size= values['imgsz1'],conf = values['conf_thres1']/100, max_det= values['max_det1'], classes= values_classes1)
            for i, pred in enumerate(result1.pred):
                if pred.shape[0]:
                    window['result1'].update('NG')
                else:
                    window['result1'].update('OK')


            img_result1 = np.squeeze(result1.render())
            img_result1 = cv2.cvtColor(img_result1, cv2.COLOR_BGR2RGB)
            imgbytes1 = cv2.imencode('.png',img_result1)[1].tobytes()
            window['image1'].update(data= imgbytes1)     

    
    if values['Continous1'] == True:
        ret, img1_org = cap1.read()
        time.sleep(set_time)
        result1 = model1(img1_org,size= values['imgsz1'],conf = values['conf_thres1']/100, max_det= values['max_det1'], classes= values_classes1)
        for i, pred in enumerate(result1.pred):
            if pred.shape[0]:
                window['result1'].update('NG')
            else:
                window['result1'].update('OK') 

        show1 = np.squeeze(result1.render()) 

        imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
        window['image1'].update(data=imgbytes1)
        # The program emulator can communicate with peripheral devices

    if event == 'file_browse1':
        window['file_weights1'].update(value=values['file_browse1'])
        window['Change1'].update(disabled=False)

    if event == 'time_continous1':
        #print(values['time_continous1'])
        set_time = values['time_continous1']/1000
        cap1 = cv2.VideoCapture(0)
        if continous1 == True:
            window['Continous1'].update(disabled=False)
    


    # webcam 2

    if event == 'Webcam2':
        cap2 = cv2.VideoCapture(1)
        recording2 = True

    elif event == 'Stop2':
        recording2 = False 
        img2 = Image.open('image.jpg')
        #img2.thumbnail((640,480))
        img2_org = img2.resize((640,480))
        imgbytes2 = ImageTk.PhotoImage(img2_org)
        window['image2'].update(data=imgbytes2)
        #window['result2'].update('')


    if event == 'classes2':
        values_classes2=[]
        for j in values['classes2']:
            values_classes2.append(CLASSES2.index(j))

    if event == 'Change2':
        model2= torch.hub.load('./levu','custom',path=values['file_weights2'],source='local',force_reload=False)
        #window['Display'].update('model2 uploaded successed')
        CLASSES2= model2.names
        window['classes2'].update(values=CLASSES2)
        if CLASSES2 is not None:
            window['Webcam2'].update(disabled= False)
            window['Detect2'].update(disabled= False)
            flag2 = True


    if recording2:
        ret, frame2 = cap2.read()
        result2 = model2(frame2,size= values['imgsz2'],conf = values['conf_thres2']/100, max_det= values['max_det2'], classes= values_classes2)


        for i, pred in enumerate(result2.pred):
            if pred.shape[0]:
                window['result2'].update('NG')
            else:
                window['result2'].update('OK')

        show2 = np.squeeze(result2.render()) 
        imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
        window['image2'].update(data=imgbytes2)


    if event == 'Snap2':
        img2_org = frame2
        #img.thumbnail((640,480))
        #img = ImageTk.PhotoImage(img)
        imgbytes2 = cv2.imencode('.png',img2_org)[1].tobytes()
        window['image2'].update(data = imgbytes2)
        recording2 = False
    if event == 'Continue2':
        recording2 = True

    # if event == 'Auto':
    #     if flag1 == True and flag2 == True:
    #         task5(model1, values['imgsz1'], values['conf_thres1']/100,max_det= values['max_det1'], classes= values_classes1)
    #         task6(model2, values['imgsz2'], values['conf_thres2']/100,max_det= values['max_det2'], classes= values_classes2)
                

    if event == 'Pic2':
        dir_img2 = sg.popup_get_file('Choose your image 2',file_types=file_name_img,keep_on_top= True)
        #print(dir_img1)
        pic2 = Image.open(dir_img2)
        img2_org = pic2.resize((640,480))
        imgbytes2 = ImageTk.PhotoImage(img2_org)
        window['image2'].update(data=imgbytes2)
        #img1 = cv2.imencode('.png',img1)[1].tobytes()
        
    if event == 'Detect2':
        result2 = model2(img2_org,size= values['imgsz2'],conf = values['conf_thres2']/100, max_det= values['max_det2'], classes= values_classes2)

        for i, pred in enumerate(result2.pred):
            if pred.shape[0]:
                window['result2'].update('NG')
            else:
                window['result2'].update('OK')

        img_result2 = np.squeeze(result2.render())
        img_result2 = cv2.cvtColor(img_result2, cv2.COLOR_BGR2RGB)
        imgbytes2 = cv2.imencode('.png',img_result2)[1].tobytes()
        window['image2'].update(data= imgbytes2)     



    # menu

    if event == 'Check Cam':
        checkcam()

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
            if event_theme == sg.WIN_CLOSED:
                break

            if event_theme == 'Apply':
                theme_choose = values_theme['theme'][0]
                window.close()
                window = make_window(theme_choose)
                #print(theme_choose)
            if event_theme == 'Cancel':
                answer = sg.popup_yes_no('Do you want to exit?')
                if answer == 'Yes':
                    break
                if answer == 'No':
                    continue
        window_theme.close()

    if event == 'Connect PLC':
        connect_PLC()


    if event == 'Check':
        dir = "D:/nc/result2/"
        file_list = os.listdir(dir)

    
    if event == 'Control':
        if recording1 == True:
            layout_control = [
                [sg.Text('Exposure',size=(10,1),font=('Helvetica',15), text_color='yellow'),
                    sg.Input(size=(10,1),font=('Helvetica',12),key='exposure',text_color='yellow',enable_events=True),
                    sg.Slider(range=(1,16777215),resolution=1,orientation='h',size=(40,20),font=('Helvetica',11), default_value=25, key= 'exposure1', enable_events=True),
                    sg.Button('OK', size=(8,1),  font=('Helvetica',10),key='choose_exposure',enable_events=True)
                    ],
                [sg.Text('Gain',size=(10,1),font=('Helvetica',15), text_color='yellow'),
                    sg.Input(size=(10,1),font=('Helvetica',12),key='gain',text_color='yellow',enable_events=True),
                    sg.Slider(range=(0,208),orientation='h',size=(40,20),font=('Helvetica',11), default_value=25, key= 'gain1', enable_events=True),
                    sg.Button('OK', size=(8,1),  font=('Helvetica',10),key='choose_gain',enable_events=True)
                    ],
                [sg.Text('Balance While Auto',size=(18,1),font=('Helvetica',15), text_color='yellow')],
                [sg.Text('Red',size=(10,1),font=('Helvetica',15), text_color='orange'),
                    sg.Input(size=(10,1),font=('Helvetica',12),key='red',text_color='orange',enable_events=True),
                    sg.Slider(range=(0,255),orientation='h',size=(40,20),font=('Helvetica',11), default_value=25, key= 'red1', enable_events=True),
                    sg.Button('OK', size=(8,1),  font=('Helvetica',10),key='choose_red',enable_events=True)
                    ],
                [sg.Text('GreenR',size=(10,1),font=('Helvetica',15), text_color='orange'),
                    sg.Input(size=(10,1),font=('Helvetica',12),key='greenr',text_color='orange',enable_events=True),
                    sg.Slider(range=(0,255),orientation='h',size=(40,20),font=('Helvetica',11), default_value=25, key= 'greenr1', enable_events=True),
                    sg.Button('OK', size=(8,1),  font=('Helvetica',10),key='choose_greenr',enable_events=True)
                    ],
                [sg.Text('GreenB',size=(10,1),font=('Helvetica',15), text_color='orange'),
                    sg.Input(size=(10,1),font=('Helvetica',12),key='greenb',text_color='orange',enable_events=True),
                    sg.Slider(range=(0,255),orientation='h',size=(40,20),font=('Helvetica',11), default_value=25, key= 'greenb1', enable_events=True),
                    sg.Button('OK', size=(8,1),  font=('Helvetica',10),key='choose_greenb',enable_events=True)
                    ],
                [sg.Text('Green',size=(10,1),font=('Helvetica',15), text_color='orange'),
                    sg.Input(size=(10,1),font=('Helvetica',12),key='green',text_color='orange',enable_events=True),
                    sg.Slider(range=(0,255),orientation='h',size=(40,20),font=('Helvetica',11), default_value=25, key= 'green1', enable_events=True),
                    sg.Button('OK', size=(8,1),  font=('Helvetica',10),key='choose_green',enable_events=True)
                    ],
                [sg.Text('Blue',size=(10,1),font=('Helvetica',15), text_color='orange'),
                    sg.Input(size=(10,1),font=('Helvetica',12),key='blue',text_color='orange',enable_events=True),
                    sg.Slider(range=(0,255),orientation='h',size=(40,20),font=('Helvetica',11), default_value=25, key= 'blue1', enable_events=True),
                    sg.Button('OK', size=(8,1),  font=('Helvetica',10),key='choose_blue',enable_events=True)
                    ],
                [sg.Text(' ')],
                [sg.Text(' '*110), sg.Button('Apply', size=(10,1),  font=('Helvetica',12),key='aplly_config',enable_events=True),sg.Button('Save', size=(10,1),  font=('Helvetica',12),key='save_config',enable_events=True)] 
            ] 
            window_control = sg.Window('Analog Control', layout_control, location=(50,50),keep_on_top=True).Finalize()
            window_control.set_min_size((500,400))   
            list_values = return_param_init()
            window_control['exposure'].update(value = list_values[0])
            window_control['exposure1'].update(value = list_values[0])
            window_control['gain'].update(value = list_values[1])
            window_control['gain1'].update(value = list_values[1])
            window_control['red'].update(value = list_values[2])
            window_control['red1'].update(value = list_values[2])
            window_control['greenr'].update(value = list_values[3])
            window_control['greenr1'].update(value = list_values[3])
            window_control['greenb'].update(value = list_values[4])
            window_control['greenb1'].update(value = list_values[4])
            window_control['green'].update(value = list_values[5])
            window_control['green1'].update(value = list_values[5])
            window_control['blue'].update(value = list_values[6])
            window_control['blue1'].update(value = list_values[6])

            while True:
                event_control, values_control = window_control.read(timeout=20)
                img1_org = my_callback.image
                img1_org = cv2.resize(img1_org,(640,480))

                if img1_org is not None:
                    show1 = img1_org
                    imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                    window['image1'].update(data=imgbytes1)

                # exposure
                if event_control == 'exposure1':
                    values_control['exposure'] = values_control['exposure1']
                    window_control['exposure'].update(value = values_control['exposure1'])
                if event_control == 'exposure':
                    values_control['exposure1'] = values_control['exposure']
                    window_control['exposure1'].update(value = values_control['exposure'])

                if event_control == 'choose_exposure':
                    if values_control['exposure'] not in range(1,16777216):
                        values_control['exposure'] = values_control['exposure1']
                        window_control['exposure'].update(value = values_control['exposure1'])
                    
                    set_exposure_or_gain("ExposureTime","ExposureTimeRaw",values_control['exposure'])
                    
                #gain 
                if event_control == 'gain1':
                    values_control['gain'] = values_control['gain1']
                    window_control['gain'].update(value = values_control['gain1'])
                if event_control == 'gain':
                    values_control['gain1'] = values_control['gain']
                    window_control['gain1'].update(value = values_control['gain'])

                if event_control == 'choose_gain':
                    if values_control['gain'] not in range(0,209):
                        values_control['gain'] = values_control['gain1']
                        window_control['gain'].update(value = values_control['gain1'])

                    set_exposure_or_gain("Gain","GainRaw",values_control['gain'])

                #BalanceWhiteAuto
                BalanceWhiteAuto('red', 'red1','choose_red',0)
                BalanceWhiteAuto('greenr', 'greenr1','choose_greenr',1)
                BalanceWhiteAuto('greenb', 'greenb1','choose_greenb',2)
                BalanceWhiteAuto('green', 'green1','choose_green',3)
                BalanceWhiteAuto('blue', 'blue1','choose_blue',4)

                #all
                if event_control == 'aplly_config':
                    set_exposure_or_gain("ExposureTime","ExposureTimeRaw",values_control['exposure1'])
                    set_exposure_or_gain("Gain","GainRaw",values_control['gain1'])

                    set_balance_white_auto(0,values_control['red1'])
                    set_balance_white_auto(1,values_control['greenr1'])
                    set_balance_white_auto(2,values_control['greenb1'])
                    set_balance_white_auto(3,values_control['green1'])
                    set_balance_white_auto(4,values_control['blue1'])


                #Save
                if event_control == 'save_config':
                    save_config(values_control['exposure1'],values_control['gain1'],values_control['red1'],values_control['greenr1'],values_control['greenb1'],values_control['green1'],values_control['blue1'])
                    sg.popup('Saved successed',font=('Helvetica',15), text_color='yellow',keep_on_top= True)
                if event_control == sg.WIN_CLOSED: 
                    break

            window_control.close()
        else:
            sg.popup("You don't open camera",font=('Helvetica',15), text_color='yellow',keep_on_top= True)



    
    if event == 'About':
        sg.popup('GUI Demo',keep_on_top= True)

st_device.acquisition_stop()
st_datastream.stop_acquisition()
window.close() 



#pyinstaller --onefile app.py yolov5/hubconf.py yolov5/models/common.py yolov5/models/experimental.py yolov5/models/yolo.py yolov5/utils/augmentations.py yolov5/utils/autoanchor.py yolov5/utils/datasets.py yolov5/utils/downloads.py yolov5/utils/general.py yolov5/utils/metrics.py yolov5/utils/plots.py yolov5/utils/torch_utils.py
#pyinstaller --onedir --windowed app.py yolov5/hubconf.py yolov5/models/common.py yolov5/models/experimental.py yolov5/models/yolo.py yolov5/utils/augmentations.py yolov5/utils/autoanchor.py yolov5/utils/datasets.py yolov5/utils/downloads.py yolov5/utils/general.py yolov5/utils/metrics.py yolov5/utils/plots.py yolov5/utils/torch_utils.py                       