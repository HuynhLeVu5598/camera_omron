
# Python program to illustrate the concept
# of threading
# importing the threading module
import threading
  
def task_camera1_snap(model,size,conf):
    pass

def task_camera2_snap(model,size,conf):
    pass

model = ''
size = 416
conf = 0.25
if __name__ == "__main__":
    #task_camera1_snap(model,size,conf)
    #task_camera2_snap(model,size,conf)
    # creating thread
    t1 = threading.Thread(target=task_camera1_snap, args=(model,size,conf,))
    t2 = threading.Thread(target=task_camera2_snap, args=(model,size,conf,))
  
    # starting thread 1
    t1.start()
    # starting thread 2
    t2.start()
  
    # wait until thread 1 is completely executed
    t1.join()
    # wait until thread 2 is completely executed
    t2.join()
  
    # both threads completely executed
    print("Done!")