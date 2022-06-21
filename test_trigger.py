import stapipy as st
import cv2
import numpy as np
import threading
import datetime 
# Image scale when displaying using OpenCV.
DISPLAY_RESIZE_FACTOR = 0.3

TRIGGER_SELECTOR = "TriggerSelector"
TRIGGER_SELECTOR_FRAME_START = "FrameStart"
TRIGGER_SELECTOR_EXPOSURE_START = "ExposureStart"
TRIGGER_MODE = "TriggerMode"
TRIGGER_MODE_ON = "On"
TRIGGER_MODE_OFF = "Off"
TRIGGER_SOURCE = "TriggerSource"
TRIGGER_SOURCE_SOFTWARE = "Software"
TRIGGER_SOFTWARE = "TriggerSoftware"

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


def datastream_callback(handle=None, context=None):

    if handle.callback_type == st.EStCallbackType.GenTLDataStreamNewBuffer:
        try:
            st_datastream = handle.module
            with st_datastream.retrieve_buffer(0) as st_buffer:
                # Check if the acquired data contains image data.
                if st_buffer.info.is_image_present:
                    # Create an image object.
                    st_image = st_buffer.get_image()
                    # Display the information of the acquired image data.
                    print("BlockID={0} Size={1} x {2} First Byte={3}".format(
                          st_buffer.info.frame_id,
                          st_image.width, st_image.height,
                          st_image.get_image_data()[0]))
                else:
                    # If the acquired data contains no image data.
                    print("Image data does not exist.")
        except st.PyStError as exception:
           print("An exception occurred.", exception)


def set_enumeration(nodemap, enum_name, entry_name):

    enum_node = st.PyIEnumeration(nodemap.get_node(enum_name))
    entry_node = st.PyIEnumEntry(enum_node[entry_name])
    # Note that depending on your use case, there are three ways to set
    # the enumeration value:
    # 1) Assign the integer value of the entry with set_int_value(val) or .value
    # 2) Assign the symbolic value of the entry with set_symbolic_value("val")
    # 3) Assign the entry (PyIEnumEntry) with set_entry_value(entry)
    # Here set_entry_value is used:
    enum_node.set_entry_value(entry_node)


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

if __name__ == "__main__":
    my_callback = CMyCallback()
    cb_func = my_callback.datastream_callback
    try:
        # Initialize StApi before using.
        st.initialize()
        # Create a system object for device scan and connection.
        st_system = st.create_system()
        # Connect to first detected device.
        st_device = st_system.create_first_device()
        # Display DisplayName of the device.
        print('Device=', st_device.info.display_name)
        nodemap = st_device.remote_port.nodemap



        # Set the TriggerSelector for FrameStart or ExposureStart.
        # try:
        #     set_enumeration(
        #         nodemap, TRIGGER_SELECTOR, TRIGGER_SELECTOR_FRAME_START)
        #     print('a')
        # except st.PyStError:
        #     set_enumeration(
        #         nodemap, TRIGGER_SELECTOR, TRIGGER_SELECTOR_EXPOSURE_START)
        #     print('b')

        # # Set the TriggerMode to On.
        # set_enumeration(nodemap, TRIGGER_MODE, TRIGGER_MODE_ON)

        # # Set the TriggerSource to Software
        # set_enumeration(nodemap, TRIGGER_SOURCE, TRIGGER_SOURCE_SOFTWARE)

        # # Get and cast to Command interface of the TriggerSoftware mode
        # trigger_software = st.PyICommand(nodemap.get_node(TRIGGER_SOFTWARE))



        # Create a datastream object for handling image stream data.
        st_datastream = st_device.create_datastream()

        # Register callback for datastream
        callback = st_datastream.register_callback(datastream_callback)

        # Start the image acquisition of the host (local machine) side.
        st_datastream.start_acquisition()

        # Start the image acquisition of the camera side.
        st_device.acquisition_start()

        while True:
            print("0 : Generate trigger")
            print("Else : Exit")
            selection = input("Select : ")
            if selection == '0':
                #trigger_software.execute()

                output_image = my_callback.image
                if output_image is not None:
                    print('c')
                    cv2.imshow('image', output_image)
                    cv2.imwrite('C:/Users/BTTB/Downloads/save_image/' + time_to_name + '.jpg')
                else:
                    print('d')
            else:
               break

        # Stop the image acquisition of the camera side
        st_device.acquisition_stop()

        # Stop the image acquisition of the host side
        st_datastream.stop_acquisition()

        # Set the TriggerMode to Off.
        set_enumeration(nodemap, TRIGGER_MODE, TRIGGER_MODE_OFF)

    except Exception as exception:
        print(exception)
