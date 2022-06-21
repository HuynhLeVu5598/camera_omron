# dont function savedata

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
import os
import datetime 
import shutil

from PIL import Image

#sys.path.append(os.path.abspath(os.path.join('../..', 'fins_omron')))
from udp import UDPFinsConnection
from initialization import FinsPLCMemoryAreas


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
    list_values = []
    with open(filename) as lines:
        for line in lines:
            _, value = line.strip().split(':')
            list_values.append(value)
    return list_values


def save_param_model1(conf,nut_me,divat,me,namchamcao,traybactruc,divatduoi,kimnamcham):
    line1 = 'conf1' + ':' + str(conf)
    line2 = 'nut_me1' + ':' + str(nut_me)
    line3 = 'divat1' + ':' + str(divat)
    line4 = 'me1' + ':' + str(me)
    line5 = 'namchamcao1' + ':' + str(namchamcao)
    line6 = 'traybactruc1' + ':' + str(traybactruc)
    line7 = 'divatduoi1' + ':' + str(divatduoi)
    line8 = 'kimnamcham1' + ':' + str(kimnamcham)

    lines = [line1,line2,line3,line4,line5,line6,line7,line8]
    with open('static/param_model1.txt', "w") as f:
        for i in lines:
            f.write(i)
            f.write('\n')

def save_param_model2(conf,nut_me,divat,me):
    line1 = 'conf' + ':' + str(conf)
    line2 = 'nut_me' + ':' + str(nut_me)
    line3 = 'divat' + ':' + str(divat)
    line4 = 'me' + ':' + str(me)

    lines = [line1,line2,line3,line4]
    with open('static/param_model2.txt', "w") as f:
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


def task1(model,size,conf):
    read_2000 = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD0\x00') # doc thanh ghi 2000
    if read_2000 ==b'\xc0\x00\x02\x00\x19\x00\x00\x01\x00`\x01\x01\x00\x00\x00%':  # gia tri 37

        directory3 = 'D:/FH/camera1/'
        if os.listdir(directory3) == []:
            print('folder 1 empty')
            #pass
        else:
            print('received folder 1')

            for filename1 in glob.glob('D:/FH/camera1/*'):
                for path1 in glob.glob(filename1 + '/*'):
                    name = path1[-18:]
                    #print(name)
                    if name == 'Input0_Camera0.jpg':
                        img1 = cv2.imread(path1)
                        while type(img1) == type(None):
                            print('loading img 1...')
                            for path1 in glob.glob(filename1 + '/*'):
                                img1 = cv2.imread(path1)

                        #name_folder_all = time_to_name()
                        #os.mkdir('D:/nc/result/Cam1/All/' + name_folder_all)
                        #cv2.imwrite('D:/nc/result/Cam1/All/' + name_folder_all  + '.jpg',img1)

                        img1 = cv2.resize(img1,(640,480))
                        result1 = model(path1,size= size,conf = conf) 

                        table1 = result1.pandas().xyxy[0]
                        area_remove = []
                        for item in range(len(table1.index)):
                            width1 = table1['xmax'][item] - table1['xmin'][item]
                            height1 = table1['ymax'][item] - table1['ymin'][item]
                            area1 = width1*height1
                            if table1['name'][item] == 'nut_me':
                                if area1 < values['area_nutme1']: 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove.append(item)

                            elif table1['name'][item] == 'divat':
                                if area1 < values['area_divat1']: 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove.append(item)

                            elif table1['name'][item] == 'me':
                                if area1 < values['area_me1']: 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove.append(item)

                            elif table1['name'][item] == 'namchamcao':
                                if height1 < values['y_namchamcao1']: 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove.append(item)

                            elif table1['name'][item] == 'tray_bac_truc':
                                if area1 < values['area_traybactruc1']: 
                                    table1.drop(item, axis=0, inplace=True)
                                    area_remove.append(item)

                        names1 = list(table1['name'])
                        print(names1)

                        len_ncc = 0
                        for i in names1:
                            if i == 'namchamcao':
                                len_ncc +=1
                        
                        save_memorys1 = []

                        if 'tray_bac_truc' in names1 or 'di_vat_duoi' in names1:
                            save_memorys1.append(1000)
                        if 'kimnamcham' not in names1:
                            save_memorys1.append(1002)
                        if 'divat' in names1 or 'me' in names1 or 'nut_me' in names1 or 'nut' in names1 or len_ncc !=2:
                            save_memorys1.append(1004)



                        display_result1 = 0
                        if 'kimnamcham' not in names1 or len_ncc !=2:
                            print('NG')
                            display_result1 = 2
                            name_folder_ng = time_to_name()
                            cv2.imwrite('F:/result/Cam1/NG/' + name_folder_ng + '.jpg',img1)

                        elif 'divat' in names1 or 'me' in names1 or 'nut_me' in names1 or 'nut' in names1:
                            print('NG')
                            display_result1 = 2
                            name_folder_ng = time_to_name()
                            cv2.imwrite('F:/result/Cam1/NG/' + name_folder_ng + '.jpg',img1)


                        elif 'tray_bac_truc' in names1 or 'di_vat_duoi' in names1:
                            print('NG')
                            display_result1 = 2
                            name_folder_ng = time_to_name()
                            cv2.imwrite('F:/result/Cam1/NG/' + name_folder_ng + '.jpg',img1)

                        else:
                            print('OK')
                            display_result1 = 1
                            #window['result1'].update(value = 'OK', text_color = 'green')
                            name_folder_ok = time_to_name()
                            cv2.imwrite('F:/result/Cam1/OK/' + name_folder_ok  + '.jpg',img1)

                        # ghi vao D2000 gia tri 0
                        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD0\x00',b'\x00\x00',1)

                        for save_memory1 in save_memorys1:
                            # bac_truc
                            if save_memory1 == 1000: 
                                # ghi vao D1000 gia tri 1 
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xE8\x00',b'\x00\x01',1)
                                # ghi vao D1006 (03EE) gia tri 2 => khong ok
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEE\x00',b'\x00\x02',1)
                            # kim nam cham
                            if save_memory1 == 1002:
                                # ghi vao D1002 gia tri 1 
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEA\x00',b'\x00\x01',1)
                                # ghi vao D1006 (03EE) gia tri 2 => khong ok
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEE\x00',b'\x00\x02',1)

                            #nam cham
                            if save_memory1 == 1004:
                                # ghi vao D1004 gia tri 1 
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEC\x00',b'\x00\x01',1)
                                # ghi vao D1006 gia tri 2 => khong ok
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEE\x00',b'\x00\x02',1)

                            #OK
                        if len(save_memorys1) == 0:
                            # ghi vao D1006 gia tri 1 
                            fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEE\x0B',b'\x00\x01',1)
                            # ghi vao D1000 gia tri 2
                            fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xE8\x00',b'\x00\x02',1)
                            # ghi vao D1002 gia tri 2
                            fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEA\x00',b'\x00\x02',1)
                            # ghi vao D1004 gia tri 2
                            fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xEC\x00',b'\x00\x02',1)             


                        show1 = np.squeeze(result1.render(area_remove))
                        show1 = cv2.resize(show1, (760,480), interpolation = cv2.INTER_AREA)
                        
                        if display_result1 == 1:
                            cv2.putText(show1, 'OK',(570,100),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        elif display_result1 == 2:
                            cv2.putText(show1, 'NG',(570,100),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                        
                        imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                        window['image1'].update(data= imgbytes1)
                        
                        print('---------------------------------------------')
 
                    if os.path.isfile(path1):
                        os.remove(path1)
                if os.path.isdir(filename1):
                    shutil.rmtree(filename1)


def task2(model,size,conf):
    read_2002 = fins_instance.memory_area_read(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD2\x00') # doc thanh ghi 2002
    if read_2002 ==b'\xc0\x00\x02\x00\x19\x00\x00\x01\x00`\x01\x01\x00\x00\x00%':  # gia tri 37
        directory4 = 'D:/FH/camera2/'
        if os.listdir(directory4) == []:
            print('folder 2 empty')
            #pass
        else:
            print('received folder 2')
            #time_initial = time.time()

            for filename2 in glob.glob('D:/FH/camera2/*'):
                for path2 in glob.glob(filename2 + '/*'):
                    #print(len(path1))
                    name = path2[-18:]
                    #print(name)
                    if name == 'Input0_Camera0.jpg':
                        img2 = cv2.imread(path2)
                        while type(img2) == type(None):
                            print('loading img 2...')
                            for path2 in glob.glob(filename2 + '/*'):
                                img2 = cv2.imread(path2)

                        #name_folder_all = time_to_name()
                        #os.mkdir('D:/nc/result/Cam2/All/' + name_folder_all)
                        #cv2.imwrite('D:/nc/result/Cam2/All/' + name_folder_all  + '.jpg',img2)

                        img2 = cv2.resize(img2,(640,480))
                        result2 = model(path2,size= size,conf = conf) 

                        table2 = result2.pandas().xyxy[0]

                        area_remove = []
                        for item in range(len(table2.index)):
                            width2 = table2['xmax'][item] - table2['xmin'][item]
                            height2 = table2['ymax'][item] - table2['ymin'][item]
                            area2 = width2*height2
                            if table2['name'][item] == 'nut_me':
                                if area2 < values['area_nutme2']: 
                                    table2.drop(item, axis=0, inplace=True)
                                    area_remove.append(item)

                            elif table2['name'][item] == 'divat':
                                if area2 < values['area_divat2']: 
                                    table2.drop(item, axis=0, inplace=True)
                                    area_remove.append(item)

                            elif table2['name'][item] == 'me':
                                if area2 < values['area_me2']: 
                                    table2.drop(item, axis=0, inplace=True)
                                    area_remove.append(item)

                        names2 = list(table2['name'])
                        print(names2)

                        save_memorys2 = []
                        if 'namcham' not in names2 or 'divat' in names2 or 'me' in names2 or 'namchamcao' in names2 or 'nut_me' in names2 or 'nut' in names2:
                            save_memorys2.append(1014)
                        # thieu kimnamcham 1012 va bactruc 1010

                        display_result2 = 0
                        if 'namcham' not in names2:
                            print('NG')
                            display_result2 = 2

                            name_folder_ng = time_to_name()
                            cv2.imwrite('F:/result/Cam2/NG/' + name_folder_ng + '.jpg',img2)
                            

                        elif 'divat' in names2 or 'me' in names2 or 'namchamcao' in names2 or 'nut_me' in names2 or 'nut' in names2:
                            print('NG')
                            display_result2 = 2

                            name_folder_ng = time_to_name()
                            cv2.imwrite('F:/result/Cam2/NG/' + name_folder_ng + '.jpg',img2)
                            

                        else:
                            print('OK')
                            display_result2 = 1

                            name_folder_ok = time_to_name()
                            cv2.imwrite('F:/result/Cam2/OK/' + name_folder_ok + '.jpg',img2)

                        # ghi vao D2002 gia tri 0
                        fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x07\xD2\x00',b'\x00\x00',1)

                        for save_memory2 in save_memorys2:
                            # bac_truc
                            if save_memory2 == 1010: 
                                # ghi vao D1010 gia tri 1 
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF2\x00',b'\x00\x01',1)
                                # ghi vao D1016 gia tri 2 => khong ok
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF8\x00',b'\x00\x02',1)
                            # kim nam cham
                            if save_memory2 == 1012:
                                # ghi vao D1012 gia tri 1 
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF4\x00',b'\x00\x01',1)
                                # ghi vao D1016 gia tri 2 => khong ok
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF8\x00',b'\x00\x02',1)

                            #nam cham
                            if save_memory2 == 1014:
                                # ghi vao D1014 gia tri 1 
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF6\x00',b'\x00\x01',1)
                                # ghi vao D1016 gia tri 2 => khong ok
                                fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF8\x00',b'\x00\x02',1)

                        #OK
                        if len(save_memorys2) == 0:
                            # ghi vao D1016 gia tri 1 
                            fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF8\x0B',b'\x00\x01',1)
                            # ghi vao D1010 gia tri 2
                            fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF2\x00',b'\x00\x02',1)
                            # ghi vao D1012 gia tri 2
                            fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF4\x00',b'\x00\x02',1)
                            # ghi vao D1014 gia tri 2
                            fins_instance.memory_area_write(FinsPLCMemoryAreas().DATA_MEMORY_WORD,b'\x03\xF6\x00',b'\x00\x02',1)             


                        show2 = np.squeeze(result2.render(area_remove))
                        show2 = cv2.resize(show2, (760,480), interpolation = cv2.INTER_AREA)

                        if display_result2 == 1:
                            cv2.putText(show2, 'OK',(570,100),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                        elif display_result2 == 2:
                            cv2.putText(show2, 'NG',(570,100),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
            
                        imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                        window['image2'].update(data= imgbytes2)
                        
                    if os.path.isfile(path2):
                        os.remove(path2)
                if os.path.isdir(filename2):
                    shutil.rmtree(filename2)
            #time_handle = time.time() - time_initial
            #print('time handle: ',time_handle)
            print('---------------------------------------------')


def make_window(theme):
    sg.theme(theme)
    
    #CLASSES1 = []
    #CLASSES2 = []

    #list_spin = [i for i in range(101)]
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
        [sg.Text('CAM 2', size =(34,1),justification='center' ,font= ('Helvetica',30),text_color='red'),
         sg.Text('CAM 1', size =(34,1),justification='center' ,font= ('Helvetica',30),text_color='red')],
        [

        # 2
        sg.Frame('',[
            #[sg.Image(filename='', size=(640,480),key='image1',background_color='black')],
            [sg.Image(filename='', size=(760,480),key='image2',background_color='black')],
            [sg.Frame('',
            [
                [sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(25,1), font=('Helvetica',12), key='file_weights2',readonly= True, text_color='navy',enable_events= True),
                sg.Frame('',[
                    [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse2',enable_events=True, disabled=True)]
                ], relief= sg.RELIEF_FLAT)],
                [sg.Text('Size', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(25,20),font=('Helvetica',11),disabled=True,default_value=416,key='imgsz2')],
                [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(25,20),font=('Helvetica',11),disabled=True,default_value=list_values2[0], key= 'conf_thres2')],
                [sg.Text('')],
                [sg.Text('')],
                [sg.Text('')],
                [sg.Text('')],
                [sg.Text('')],
                #[sg.Text('Nut me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,8000),orientation='h',size=(25,20),font=('Helvetica',11),disabled=True,default_value=list_values2[1], resolution=5, key= 'area_nutme2')],
                #[sg.Text('Di vat',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,8000),orientation='h',size=(25,20),font=('Helvetica',11),disabled=True,default_value=list_values2[2], resolution=5, key= 'area_divat2')],
                #[sg.Text('Me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,8000),orientation='h',size=(25,20),font=('Helvetica',11),disabled=True ,default_value=list_values2[3], resolution=5, key= 'area_me2')],
                #[sg.Text('Result1',size=(12,5),font=('Helvetica',15),text_color='yellow',expand_y =True),sg.InputText('',size=(16,5),justification='center',font=('Helvetica',30),text_color='red',readonly=True, key='result1',expand_y=True)]
            ], vertical_alignment='top'),
            sg.Frame('',[
                #[sg.Text('')],
                [sg.Button('Webcam', size=(8,1),  font=('Helvetica',14),disabled=True,key= 'Webcam2')],
                [sg.Text('')],
                [sg.Button('Stop', size=(8,1), font=('Helvetica',14),disabled=True  ,key='Stop2')],
                [sg.Text('')],
                [sg.Button('Continue', size=(8,1),  font=('Helvetica',14),disabled=True ,key='Continue2')],
                [sg.Text('')],
                [sg.Button('Snap', size=(8,1), font=('Helvetica',14),disabled=True  ,key='Snap2')],
                #[sg.Text('')],

                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT),

            sg.Frame('',[   
                [sg.Button('Change', size=(8,1), font=('Helvetica',14), disabled= True, key= 'Change2')],
                [sg.Text('')],
                [sg.Button('Pic', size=(8,1), font=('Helvetica',14),disabled=True,key='Pic2')],
                [sg.Text('')],
                [sg.Button('Detect', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'Detect2')],
                [sg.Text('')],
                [sg.Button('SaveData', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'SaveData2')],

                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT),
            ]
        ], expand_y= True),

        #1
        sg.Frame('',[
            #[sg.Image(filename='', size=(640,480),key='image1',background_color='black')],
            [sg.Image(filename='', size=(760,480),key='image1',background_color='black')],
            [sg.Frame('',
            [
                [sg.Text('Weights', size=(12,1), font=('Helvetica',15),text_color='yellow'), sg.Input(size=(25,1), font=('Helvetica',12), key='file_weights1',readonly= True, text_color='navy',enable_events= True),
                sg.Frame('',[
                    [sg.FileBrowse(file_types= file_weights, size=(12,1), font=('Helvetica',10),key= 'file_browse1',enable_events=True,disabled=True )]
                ], relief= sg.RELIEF_FLAT)],
                [sg.Text('Size', size=(12,1),font=('Helvetica',15), text_color='yellow'),sg.InputCombo((416,512,608,896,1024,1280,1408,1536),size=(25,20),font=('Helvetica',11),disabled=True ,default_value=416,key='imgsz1')],
                [sg.Text('Confidence',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,100),orientation='h',size=(25,20),font=('Helvetica',11),disabled=True  ,default_value=list_values1[0], key= 'conf_thres1')],
                [sg.Text('')],
                [sg.Text('')],
                [sg.Text('')],
                [sg.Text('')],
                [sg.Text('')],
                #[sg.Text('Nut me',size=(12,1),font=('Helvetica',12), text_color='yellow'), sg.Slider(range=(1,4000),orientation='h',size=(25,10),font=('Helvetica',11), disabled=True ,default_value=list_values1[1], resolution=5, key= 'area_nutme1')],
                #[sg.Text('Di vat',size=(12,1),font=('Helvetica',12), text_color='yellow'), sg.Slider(range=(1,4000),orientation='h',size=(25,10),font=('Helvetica',11), disabled=True ,default_value=list_values1[2], resolution=5, key= 'area_divat1')],
                #[sg.Text('Me',size=(12,1),font=('Helvetica',12), text_color='yellow'), sg.Slider(range=(1,4000),orientation='h',size=(25,10),font=('Helvetica',11), disabled=True ,default_value=list_values1[3], resolution=5, key= 'area_me1')],
                #[sg.Text('Nam cham cao',size=(12,1),font=('Helvetica',12), text_color='yellow'), sg.Slider(range=(1,200),orientation='h',size=(25,10),font=('Helvetica',11), disabled=True ,default_value=list_values1[4], resolution=1, key= 'y_namchamcao1')],
                #[sg.Text('Tray bac truc',size=(12,1),font=('Helvetica',12), text_color='yellow'), sg.Slider(range=(1,4000),orientation='h',size=(25,10),font=('Helvetica',11), disabled=True ,default_value=list_values1[5], resolution=5, key= 'area_traybactruc1')],
                #[sg.Text('Di vat duoi',size=(12,1),font=('Helvetica',12), text_color='yellow'), sg.Slider(range=(1,4000),orientation='h',size=(25,10),font=('Helvetica',11), disabled=True ,default_value=list_values1[6], resolution=5, key= 'area_divatduoi1')],
                #[sg.Text('Kim nam cham',size=(12,1),font=('Helvetica',12), text_color='yellow'), sg.Slider(range=(1,200),orientation='h',size=(25,10),font=('Helvetica',11), disabled=True ,default_value=list_values1[7], resolution=1, key= 'y_kimnamcham1')],
                #[sg.Text('Result1',size=(12,5),font=('Helvetica',15),text_color='yellow',expand_y =True),sg.InputText('',size=(16,5),justification='center',font=('Helvetica',30),text_color='red',readonly=True, key='result1',expand_y=True)]
            ], vertical_alignment='top'),
            sg.Frame('',[
                #[sg.Text('')],
                [sg.Button('Webcam', size=(8,1),  font=('Helvetica',14),disabled=True  ,key= 'Webcam1')],
                [sg.Text('')],
                [sg.Button('Stop', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'Stop1')],
                [sg.Text('')],
                [sg.Button('Continue', size=(8,1),  font=('Helvetica',14), disabled=True, key= 'Continue1')],
                [sg.Text('')],
                [sg.Button('Snap', size=(8,1), font=('Helvetica',14),disabled=True ,key= 'Snap1')],
                [sg.Text('')],

                #],element_justification='center',expand_x=True, vertical_alignment='top', relief= sg.RELIEF_FLAT),
                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT),

            sg.Frame('',[   
                [sg.Button('Change', size=(8,1), font=('Helvetica',14), disabled= True, key= 'Change1')],
                [sg.Text('')],
                [sg.Button('Pic', size=(8,1), font=('Helvetica',14),disabled=True,key= 'Pic1')],
                [sg.Text('')],
                [sg.Button('Detect', size=(8,1), font=('Helvetica',14),disabled=True,key= 'Detect1')],
                [sg.Text('')],
                [sg.Button('SaveData', size=(8,1), font=('Helvetica',14),disabled=True,key= 'SaveData1')],

                ],element_justification='center', vertical_alignment='top', relief= sg.RELIEF_FLAT),
            ]
        ], expand_y= True),
    
    ]] 

    layout_parameter = [
        [
        # 2
        sg.Frame('',[
            [sg.Frame('',
            [
                [sg.Text('Nut me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,8000),orientation='h',size=(35,20),font=('Helvetica',11),disabled=True,default_value=list_values2[1], resolution=5, key= 'area_nutme2')],
                [sg.Text('Di vat',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,8000),orientation='h',size=(35,20),font=('Helvetica',11),disabled=True,default_value=list_values2[2], resolution=5, key= 'area_divat2')],
                [sg.Text('Me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,8000),orientation='h',size=(35,20),font=('Helvetica',11),disabled=True ,default_value=list_values2[3], resolution=5, key= 'area_me2')],
                #[sg.Text('Result1',size=(12,5),font=('Helvetica',15),text_color='yellow',expand_y =True),sg.InputText('',size=(16,5),justification='center',font=('Helvetica',30),text_color='red',readonly=True, key='result1',expand_y=True)]
            ]),
            ]
        ], expand_y= True),

        #1
        sg.Frame('',[
            [sg.Frame('',
            [
                [sg.Text('Nut me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,4000),orientation='h',size=(35,20),font=('Helvetica',11), disabled=True ,default_value=list_values1[1], resolution=5, key= 'area_nutme1')],
                [sg.Text('Di vat',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,4000),orientation='h',size=(35,20),font=('Helvetica',11), disabled=True ,default_value=list_values1[2], resolution=5, key= 'area_divat1')],
                [sg.Text('Me',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,4000),orientation='h',size=(35,20),font=('Helvetica',11), disabled=True ,default_value=list_values1[3], resolution=5, key= 'area_me1')],
                [sg.Text('Nam cham cao',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,200),orientation='h',size=(35,20),font=('Helvetica',11), disabled=True ,default_value=list_values1[4], resolution=1, key= 'y_namchamcao1')],
                [sg.Text('Tray bac truc',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,4000),orientation='h',size=(35,20),font=('Helvetica',11), disabled=True ,default_value=list_values1[5], resolution=5, key= 'area_traybactruc1')],
                [sg.Text('Di vat duoi',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,4000),orientation='h',size=(35,20),font=('Helvetica',11), disabled=True ,default_value=list_values1[6], resolution=5, key= 'area_divatduoi1')],
                [sg.Text('Kim nam cham',size=(12,1),font=('Helvetica',15), text_color='yellow'), sg.Slider(range=(1,200),orientation='h',size=(35,20),font=('Helvetica',11), disabled=True ,default_value=list_values1[7], resolution=1, key= 'y_kimnamcham1')]
                #[sg.Text('Result1',size=(12,5),font=('Helvetica',15),text_color='yellow',expand_y =True),sg.InputText('',size=(16,5),justification='center',font=('Helvetica',30),text_color='red',readonly=True, key='result1',expand_y=True)]
            ]),
            ]
        ], expand_y= True),
    
    ]] 


    layout_terminal = [[sg.Text("Anything printed will display here!")],
                      [sg.Multiline( font=('Helvetica',14), expand_x=True, expand_y=True, write_only=True, autoscroll=True, auto_refresh=True,reroute_stdout=True, reroute_stderr=True, echo_stdout_stderr=True)]
                      ]
    
    layout = [[sg.TabGroup([[  sg.Tab('Main', layout_main),
                               sg.Tab('Parameter', layout_parameter),
                               sg.Tab('Output', layout_terminal)]], expand_x=True, expand_y=True)
               ]]


    window = sg.Window('HuynhLeVu', layout, location=(0,0),right_click_menu=right_click_menu).Finalize()
    window.Maximize()

    return window




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

list_values1 = load_param_model('static/param_model1.txt')
list_values2 = load_param_model('static/param_model2.txt')


themes = load_theme()
theme = themes[0]
#theme = 'Purple'
#window,img1_org, img2_org = make_window(theme)
window = make_window(theme)

values_classes1=[]
values_classes2=[]

# connected = False
# while connected == False:
#     connected = connect_plc('192.168.250.1')
#     print('connecting ....')
#     event, values = window.read(timeout=20)

# print("connected plc")   

#"C:\Users\Administrator\Documents\4\fins_omron\fins\best1_h.pt"
mypath1 = "D:/Connect/4/fins_omron/fins/best2_h.pt"
model1 = torch.hub.load('./levu','custom', path= mypath1, source='local',force_reload =False)
print('model1 already')

mypath2 = "D:/Connect/4/fins_omron/fins/best2_v.pt"
model2 = torch.hub.load('./levu','custom', path= mypath2, source='local',force_reload =False)
print('model2 already')


# window['Webcam1'].update(disabled= True)
# window['Webcam2'].update(disabled= True)
# window['Stop1'].update(disabled= True)
# window['Stop2'].update(disabled= True)
# window['Continue1'].update(disabled= True)
# window['Continue2'].update(disabled= True)
# window['Snap1'].update(disabled= True)
# window['Snap2'].update(disabled= True)
# window['Change1'].update(disabled= True)
# window['Change2'].update(disabled= True)
# window['Pic1'].update(disabled= True)
# window['Pic2'].update(disabled= True)
# window['Detect1'].update(disabled= True)
# window['Detect2'].update(disabled= True)
# window['SaveData1'].update(disabled= True)
# window['SaveData2'].update(disabled= True)



#model1 = model
#model2 = model


#size = 416
#conf = 0.5
#myarea = 700
#max_det=1000
#classes = 0

removefile()



while True:
    event, values = window.read(timeout=20)

    #task1(model1,size= values['imgsz1'],conf= values['conf_thres1']/100)
    #task2(model2,size= values['imgsz2'],conf= values['conf_thres2']/100) 

    #task1(model,size,conf)
    #task2(model,size,conf) 


    # menu
    if event =='Exit' or event == sg.WIN_CLOSED:
        break

    if event =='Administrator':
        login_password = 'vu123'  # helloworld
        password = sg.popup_get_text(
            'Enter Password: ', password_char='*') 
        if password == login_password:
            sg.popup_ok('Login Successed!!! ',text_color='green', font=('Helvetica',14))  

            window['imgsz2'].update(disabled= False)
            window['imgsz1'].update(disabled= False)
            window['conf_thres2'].update(disabled= False)
            window['conf_thres1'].update(disabled= False)
            window['area_nutme2'].update(disabled= False)
            window['area_nutme1'].update(disabled= False)
            window['area_divat2'].update(disabled= False)
            window['area_divat1'].update(disabled= False)
            window['area_me2'].update(disabled= False)
            window['area_me1'].update(disabled= False)
            window['y_namchamcao1'].update(disabled= False)
            window['area_traybactruc1'].update(disabled= False)
            window['area_divatduoi1'].update(disabled= False)
            window['y_kimnamcham1'].update(disabled= False)

            window['file_browse2'].update(disabled= False,button_color='turquoise')
            window['file_browse1'].update(disabled= False,button_color='turquoise')

            window['Stop1'].update(disabled= False,button_color='turquoise')
            window['Stop2'].update(disabled= False,button_color='turquoise')
            window['Pic1'].update(disabled= False,button_color='turquoise')
            window['Pic2'].update(disabled= False,button_color='turquoise')
            window['SaveData1'].update(disabled= False,button_color='turquoise')
            window['SaveData2'].update(disabled= False,button_color='turquoise')
            window['Webcam1'].update(button_color='turquoise')
            window['Webcam2'].update(button_color='turquoise')
            window['Continue1'].update(button_color='turquoise')
            window['Continue2'].update(button_color='turquoise')
            window['Snap1'].update(button_color='turquoise')
            window['Snap2'].update(button_color='turquoise')
            window['Change1'].update(button_color='turquoise')
            window['Change2'].update(button_color='turquoise')
            window['Detect1'].update(button_color='turquoise')
            window['Detect2'].update(button_color='turquoise')
 
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
            if event_theme == sg.WIN_CLOSED:
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



    # webcam 1

    if event == 'Webcam1':
        cap1 = cv2.VideoCapture(0)
        recording1 = True
        window['Continue1'].update(disabled=False)

    elif event == 'Stop1':
        recording1 = False 
        imgbytes1 = np.zeros([100,100,3],dtype=np.uint8)
        imgbytes1 = cv2.resize(imgbytes1, (760,480), interpolation = cv2.INTER_AREA)
        imgbytes1 = cv2.imencode('.png',imgbytes1)[1].tobytes()
        window['image1'].update(data=imgbytes1)

    if event == 'Change1':
        model1= torch.hub.load('./levu','custom',path=values['file_weights1'],source='local',force_reload=False)
        CLASSES1= model1.names
        continous1 = True
        flag_listbox = True
        have_model1 = True

        if CLASSES1 is not None:
            window['Detect1'].update(disabled= False)
            flag1 = True


    # if recording1:
    #     if have_model1 == True:
    #         ret, img1_org = cap1.read()

    #         result1 = model1(img1_org,size= values['imgsz1'],conf = values['conf_thres1']/100)

    #         for i, pred in enumerate(result1.pred):
    #             if pred.shape[0]:
    #                 window['result1'].update('NG')
    #             else:
    #                 window['result1'].update('OK')


    #         show1 = np.squeeze(result1.render()) 
    #         imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
    #         window['image1'].update(data=imgbytes1)
    #     else:
    #         ret, img1_org = cap1.read()
    #         show1 = img1_org
    #         imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
    #         window['image1'].update(data=imgbytes1)

    #if event == 'Snap1':
        #window['image1'].update(background_color='black')
        #img1_org = frame1
        #img.thumbnail((640,480))
        #img = ImageTk.PhotoImage(img)
        #if recording1 == True:
        #    imgbytes1 = cv2.imencode('.png',img1_org)[1].tobytes()
        #    window['image1'].update(data = imgbytes1)
        #    recording1 = False

    if event == 'Continue1':
        recording1 = True

    if event == 'Pic1':
        dir_img1 = sg.popup_get_file('Choose your image 1',file_types=file_name_img,keep_on_top= True)
        if dir_img1 not in ('',None):
            pic1 = Image.open(dir_img1)
            #img1_org = pic1.resize((640,480))
            img1_org = pic1.resize((760,480))
            imgbytes1 = ImageTk.PhotoImage(img1_org)
            window['image1'].update(data= imgbytes1)
            #window['Detect1'].update(disabled= False)


    if event == 'Detect1':
        if have_model1 == True:
            try:
                result1 = model1(dir_img1,size= values['imgsz1'],conf = values['conf_thres1']/100)
                table1 = result1.pandas().xyxy[0]

                area_remove = []
                for item in range(len(table1.index)):
                    width1 = table1['xmax'][item] - table1['xmin'][item]
                    height1 = table1['ymax'][item] - table1['ymin'][item]
                    area1 = width1*height1

                    if table1['name'][item] == 'nut_me':
                        if area1 < values['area_nutme1']: 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove.append(item)

                    elif table1['name'][item] == 'divat':
                        if area1 < values['area_divat1']: 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove.append(item)

                    elif table1['name'][item] == 'me':
                        if area1 < values['area_me1']: 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove.append(item)

                    elif table1['name'][item] == 'namchamcao':
                        if height1 < values['y_namchamcao1']: 
                            table1.drop(item, axis=0, inplace=True)
                            area_remove.append(item)

                names1 = list(table1['name'])

                len_ncc = 0
                for i in names1:
                    if i == 'namchamcao':
                        len_ncc +=1

                display_result1 = 0
                if 'kimnamcham' not in names1 or len_ncc !=2:
                    print('NG')
                    display_result1 = 2

                elif 'divat' in names1 or 'me' in names1 or 'nut_me' in names1 or 'nut' in names1:
                    print('NG')
                    display_result1 = 2

                elif 'tray_bac_truc' in names1 or 'di_vat_duoi' in names1:
                    print('NG')
                    display_result1 = 2

                else:
                    print('OK')
                    display_result1 = 1
        
                show1 = np.squeeze(result1.render(area_remove))
                show1 = cv2.resize(show1, (760,480), interpolation = cv2.INTER_AREA)
                if display_result1 == 1:
                    #cv2.putText(show1, 'OK',(450,120),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                    cv2.putText(show1, 'OK',(570,100),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                elif display_result1 == 2:
                    cv2.putText(show1, 'NG',(570,100),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
                

                imgbytes1 = cv2.imencode('.png',show1)[1].tobytes()
                window['image1'].update(data= imgbytes1)
            
            except:
                sg.popup_annoying("Don't have image", font=('Helvetica',14),text_color='red')



    if event == 'file_browse1':
        window['file_weights1'].update(value=values['file_browse1'])
        window['Change1'].update(disabled=False)

    if event == 'time_continous1':
        #print(values['time_continous1'])
        set_time = values['time_continous1']/1000
        cap1 = cv2.VideoCapture(0)
        if continous1 == True:
            window['Continous1'].update(disabled=False)

    if event == 'SaveData1':
        save_param_model1(values['conf_thres1'],values['area_nutme1'], values['area_divat1'],values['area_me1'],values['y_namchamcao1'],values['area_traybactruc1'],values['area_divatduoi1'] ,values['y_kimnamcham1'])
    

    # webcam 2

    if event == 'Webcam2':
        cap2 = cv2.VideoCapture(1)
        recording2 = True

    elif event == 'Stop2':
        recording2 = False 
        imgbytes2 = np.zeros([100,100,3],dtype=np.uint8)
        imgbytes2 = cv2.resize(imgbytes2, (760,480), interpolation = cv2.INTER_AREA)
        imgbytes2 = cv2.imencode('.png',imgbytes2)[1].tobytes()
        window['image2'].update(data=imgbytes2)


    if event == 'Change2':
        model2= torch.hub.load('./levu','custom',path=values['file_weights2'],source='local',force_reload=False)
        CLASSES2= model2.names
        continous2 = True
        flag_listbox = True
        have_model2 = True

        if CLASSES2 is not None:
            window['Detect2'].update(disabled= False)
            flag2 = True


    # if recording2:
    #     ret, frame2 = cap2.read()
    #     result2 = model2(frame2,size= values['imgsz2'],conf = values['conf_thres2']/100)


    #     for i, pred in enumerate(result2.pred):
    #         if pred.shape[0]:
    #             window['result2'].update('NG')
    #         else:
    #             window['result2'].update('OK')

    #     show2 = np.squeeze(result2.render()) 
    #     imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
    #     window['image2'].update(data=imgbytes2)


    #if event == 'Snap2':
        #window['image2'].update(background_color='black')
        #img2_org = frame2
        #img.thumbnail((640,480))
        #img = ImageTk.PhotoImage(img)
        #imgbytes2 = cv2.imencode('.png',img2_org)[1].tobytes()
        #window['image2'].update(data = imgbytes2)
        #recording2 = False
        
    if event == 'Continue2':
        recording2 = True



    if event == 'file_browse2':
        window['file_weights2'].update(value=values['file_browse2'])
        window['Change2'].update(disabled=False)
                

    if event == 'Pic2':
        dir_img2 = sg.popup_get_file('Choose your image 2',file_types=file_name_img,keep_on_top= True)
        if dir_img2 not in ('',None):
            pic2 = Image.open(dir_img2)
            img2_org = pic2.resize((760,480))
            imgbytes2 = ImageTk.PhotoImage(img2_org)
            window['image2'].update(data=imgbytes2)


        
    if event == 'Detect2':
        if have_model2 == True:
            try:
                result2 = model2(dir_img2,size= values['imgsz2'],conf = values['conf_thres2']/100)

                table2 = result2.pandas().xyxy[0]

                area_remove = []
                for item in range(len(table2.index)):
                    width2 = table2['xmax'][item] - table2['xmin'][item]
                    height2 = table2['ymax'][item] - table2['ymin'][item]
                    area2 = width2*height2
                    if table2['name'][item] == 'nut_me':
                        if area2 < values['area_nutme2']: 
                            table2.drop(item, axis=0, inplace=True)
                            area_remove.append(item)
                    
                    elif table2['name'][item] == 'divat':
                        if area2 < values['area_divat2']: 
                            table2.drop(item, axis=0, inplace=True)
                            area_remove.append(item)

                    elif table2['name'][item] == 'me':
                        if area2 < values['area_me2']: 
                            table2.drop(item, axis=0, inplace=True)
                            area_remove.append(item)

                names = list(table2['name'])

                display_result = 0
                if 'namcham' not in names:
                    print('NG')
                    display_result = 2
                    #window['result2'].update(value = 'NG', text_color = 'red')
                elif 'divat' in names or 'me' in names or 'namchamcao' in names or 'nut_me' in names or 'nut' in names:
                    print('NG')
                    display_result = 2
                    #window['result2'].update(value = 'NG', text_color = 'red')
                else:
                    print('OK')
                    display_result = 1
                    #window['result2'].update(value = 'OK', text_color = 'green')


                show2 = np.squeeze(result2.render(area_remove))
                show2 = cv2.resize(show2, (760,480), interpolation = cv2.INTER_AREA)

                if display_result == 1:
                    cv2.putText(show2, 'OK',(570,100),cv2.FONT_HERSHEY_COMPLEX, 4,(0,255,0),5)
                elif display_result == 2:
                    cv2.putText(show2, 'NG',(570,100),cv2.FONT_HERSHEY_COMPLEX, 4,(0,0,255),5)
    
                imgbytes2 = cv2.imencode('.png',show2)[1].tobytes()
                window['image2'].update(data= imgbytes2)
            except:
                sg.popup_annoying("Don't have image", font=('Helvetica',14),text_color='red')
    
    if event == 'SaveData2':
        save_param_model2(values['conf_thres2'] ,values['area_nutme2'], values['area_divat2'],values['area_me2'])


window.close() 


#pyinstaller --onefile app.py yolov5/hubconf.py yolov5/models/common.py yolov5/models/experimental.py yolov5/models/yolo.py yolov5/utils/augmentations.py yolov5/utils/autoanchor.py yolov5/utils/datasets.py yolov5/utils/downloads.py yolov5/utils/general.py yolov5/utils/metrics.py yolov5/utils/plots.py yolov5/utils/torch_utils.py
#pyinstaller --onedir --windowed app.py yolov5/hubconf.py yolov5/models/common.py yolov5/models/experimental.py yolov5/models/yolo.py yolov5/utils/augmentations.py yolov5/utils/autoanchor.py yolov5/utils/datasets.py yolov5/utils/downloads.py yolov5/utils/general.py yolov5/utils/metrics.py yolov5/utils/plots.py yolov5/utils/torch_utils.py                       