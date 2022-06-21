"""
 This sample shows how to use trigger mode of the camera
 The The following points will be demonstrated in this sample code:
 - Initialize StApi
 - Connect to camera
 - Set trigger mode and send trigger
"""

import stapipy as st

TRIGGER_SELECTOR = "TriggerSelector"
TRIGGER_SELECTOR_FRAME_START = "FrameStart"
TRIGGER_SELECTOR_EXPOSURE_START = "ExposureStart"
TRIGGER_MODE = "TriggerMode"
TRIGGER_MODE_ON = "On"
TRIGGER_MODE_OFF = "Off"
TRIGGER_SOURCE = "TriggerSource"
TRIGGER_SOURCE_SOFTWARE = "Software"
TRIGGER_SOFTWARE = "TriggerSoftware"

def datastream_callback(handle=None, context=None):
    """
    Callback to handle events from DataStream.

    :param handle: handle that trigger the callback.
    :param context: user data passed on during callback registration.
    """
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
    """
    Function to set enumeration value.

    :param nodemap: node map.
    :param enum_name:  name of the enumeration node.
    :param entry_name:  symbolic value of the enumeration entry node.
    """
    enum_node = st.PyIEnumeration(nodemap.get_node(enum_name))
    entry_node = st.PyIEnumEntry(enum_node[entry_name])
    # Note that depending on your use case, there are three ways to set
    # the enumeration value:
    # 1) Assign the integer value of the entry with set_int_value(val) or .value
    # 2) Assign the symbolic value of the entry with set_symbolic_value("val")
    # 3) Assign the entry (PyIEnumEntry) with set_entry_value(entry)
    # Here set_entry_value is used:
    enum_node.set_entry_value(entry_node)


if __name__ == "__main__":
    try:
        # Initialize StApi before using.
        st.initialize()

        # Create a system object for device scan and connection.
        st_system = st.create_system()

        # Connect to first detected device.
        st_device = st_system.create_first_device()

        st_device2 = st_system.create_first_device()

        # Display DisplayName of the device.
        print('Device=', st_device.info.display_name)
        print('Device2=', st_device2.info.display_name)

        # Get the nodemap for the camera settings.
        nodemap = st_device.remote_port.nodemap

        # Set the TriggerSelector for FrameStart or ExposureStart.
        try:
            set_enumeration(
                nodemap, TRIGGER_SELECTOR, TRIGGER_SELECTOR_FRAME_START)
        except st.PyStError:
            set_enumeration(
                nodemap, TRIGGER_SELECTOR, TRIGGER_SELECTOR_EXPOSURE_START)

        # Set the TriggerMode to On.
        set_enumeration(nodemap, TRIGGER_MODE, TRIGGER_MODE_ON)

        # Set the TriggerSource to Software
        set_enumeration(nodemap, TRIGGER_SOURCE, TRIGGER_SOURCE_SOFTWARE)

        # Get and cast to Command interface of the TriggerSoftware mode
        trigger_software = st.PyICommand(nodemap.get_node(TRIGGER_SOFTWARE))

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
                trigger_software.execute()
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
