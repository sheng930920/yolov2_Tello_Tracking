import time
import cv2
from djitellopy import Tello

#connect drone
drone = Tello()
drone.connect()
drone.streamon()
drone.takeoff()


def move_forward(retry):
    i = 0
    while 1:
        cv2.imshow('Video', drone.get_frame_read().frame)
        time.sleep(1)
        while i < retry:
            drone.move_forward(500)
            time.sleep(3)#for step by step movement
            i += 1
        drone.land()
        if cv2.waitKey(10) & 0xFF == ord('q'):
            drone.end()
            break

def main():
    move_forward(8)#use count for determine total move forward value'8*500=40 metres'


if __name__ == '__main__':
    main()
