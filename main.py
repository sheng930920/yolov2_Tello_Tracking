from djitellopy import Tello
import tkinter as tk
from threading import Thread
import time
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from darkflow.net.build import TFNet

drone = Tello()
options = {"model": "cfg/yolov2-custom-ozan.cfg",
           "load": "ckpt/yolov2-custom-ozan_best70.weights",
           "gpu": 0.8,
           "threshold": 0.4
           }  # determine configs
yolo_net = TFNet(options)

global previouse_box, target, FPS, output  # kodun her bölgesinde kullanılacak değişkenlenlerin tanımlanması
previouse_box = None
FPS = None
# noinspection SpellCheckingInspection,PyUnusedLocal
class Detector:

    def boxing(self, originalimage,
               predictions):  #draw boundary box
        global right_left_velocity, forw_back_velocity, up_down_velocity, yaw_velocity, previouse_box
        newImage = np.copy(originalimage)
        right_left_velocity, forw_back_velocity, up_down_velocity, yaw_velocity = None, None, None, None

        for result in predictions:  # gelen tahminleri işleyerek her birinin etiket ve konum verilerinin işlenmesi
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']

            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']

            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3))
            takip = result['label']

            if confidence > 0.4 and takip == Detector.takipteki(self,
                                                                2):  # check confidence value and target class for tracking
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)
                newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                       (0, 230, 0), 1, cv2.LINE_AA)
                ground_truth = (top_x, top_y, btm_x, btm_y)
                target_x = top_x + (btm_x - top_x) / 2
                target_y = top_y + (btm_y - top_y) / 2

                if self.previouse_box is None:  # store first boundary box coordinates
                    self.previouse_box = (top_x, top_y, btm_x, btm_y)
                iou = Detector.bb_intersection_over_union(self, self.previouse_box, ground_truth) #calculate IOU for continue tracking first object that in video
                print(f'Relationship Between Previous and Current Boxs={iou}')
                if iou > 0.4:  # update previous box values for tracking 
                    self.previouse_box = (top_x, top_y, btm_x, btm_y)
                    cv2.arrowedLine(newImage, (480, 320), (int(target_x), int(target_y)), (0, 255, 255), 2)#draw arrow for object that tracking
                    distance_x = target_x - 480 # calculate distance with difference with pixel values
                    distancey = 360 - target_y
                    if Detector.takipteki(self, 2) == "insan":# determine speed according to target class
                        forw_back_velocity = distancey / 2
                        right_left_velocity = distance_x / 4
                        if -40 <= distance_x <= 40:
                            right_left_velocity = 0
                        if -40 <= distancey <= 40:
                            forw_back_velocity = 0
                        yaw_velocity = right_left_velocity / 4

                    if Detector.takipteki(self, 2) == "araba":
                        forw_back_velocity = distancey / 2
                        right_left_velocity = distance_x / 4
                        yaw_velocity = right_left_velocity / 2
                        if -50 <= distance_x <= 50:
                            right_left_velocity = 0
                            yaw_velocity = 0
                        if -40 <= distancey <= 40:
                            forw_back_velocity = 0
                    up_down_velocity = 0
                    drone.send_rc_control(right_left_velocity, forw_back_velocity, 0, yaw_velocity, 1)
                    print(
                        f'Yanal Hızı: {right_left_velocity}, Boylamsal Hızı: {up_down_velocity}, Dönüş Hızı: {yaw_velocity},Uzamsal Hızı:{forw_back_velocity}')
        return newImage

    def takipteki(self, i):#reset previous box and reset speed when changed tracking class
        if i == 1:
            time.sleep(0.1)
            drone.send_rc_control(0, 0, 0, 0, 1)
            self.previouse_box = None
            self.target = "insan"
            return self.target
        if i == 0:
            drone.send_rc_control(0, 0, 0, 0, 1)
            self.previouse_box = None
            self.target = "araba"
            return self.target
        if i != 0 and i != 1:
            return self.target

    def stop(self):
        drone.land()

    def bb_intersection_over_union(self, box_a, box_b):

        xA = max(box_a[0], box_b[0])
        yA = max(box_a[1], box_b[1])
        xB = min(box_a[2], box_b[2])
        yB = min(box_a[3], box_b[3])
        interArea = (xB - xA) * (yB - yA)
        boxAArea = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        boxBArea = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def video_detect(self):
        self.previouse_box = None
        self.target = "insan"
        while 1:
            image = drone.get_frame_read().frame
            start_time = time.time()
            original_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = yolo_net.return_predict(original_img)
            image = Detector.boxing(self, image, results)
            FPS = 1.0 / (time.time() - start_time)
            info = "Takip Edilen Sinif::" + " " + str(Detector.takipteki(self, 2).capitalize())
            if cv2.waitKey(1) == ord("a"):  # change track class to car
                Detector.takipteki(self, 0)
            if cv2.waitKey(1) == ord("i"):  # change track class to pedestrian
                Detector.takipteki(self, 1)
            if cv2.waitKey(1) == ord("e"):  # emergency stop engine 
                Detector.stop(self)
            if cv2.waitKey(1) == ord("q"):  # exit tracking mode
                drone.send_rc_control(0, 0, 0, 0, 1)
                cv2.destroyAllWindows()
                break
            if FPS is not None:
                drone_state = [drone.get_battery(), drone.get_flight_time()]
                state_text = "Battery State:" + "  " + str(
                    drone_state[0]) + "     " + "Flight Time:" + " " + str(drone_state[1])
                image = cv2.putText(image, str(FPS), (890, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (222, 255, 8), 1,
                                    cv2.LINE_AA)
                if drone_state[0] < 20:
                    image = cv2.putText(image, str(state_text), (20, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                        (41, 0, 253), 1,
                                        cv2.LINE_AA)
                if drone_state[0] >= 20:
                    image = cv2.putText(image, str(state_text), (20, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                        (8, 253, 41), 1,
                                        cv2.LINE_AA)
                image = cv2.putText(image, str(info), (600, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (8, 253, 41), 1,
                                    cv2.LINE_AA)
            cv2.imshow('Image', image)


class pencere(tk.Tk):
    def __init__(self):
        super().__init__()

        self.connect = tk.Button(text="Connect Drone", command=self.connection)
        self.connect.place(relx=0.3, rely=0.85, relwidth=0.4)
        self.connect.config(font=("Times New Roman", 12, "bold"))
        self.connect.config(fg="blue")

        self.havalan = tk.Button(text="Takeoff", command=self.havalan)
        self.havalan.place(relx=0.03, rely=0.85, relwidth=0.25)
        self.havalan.config(font=("Times New Roman", 12, "bold"))
        self.havalan.config(fg="red2")

        self.land = tk.Button(text="Land", command=drone.land)
        self.land.place(relx=0.72, rely=0.85, relwidth=0.25)
        self.land.config(font=("Times New Roman", 12, "bold"))
        self.land.config(fg="green2")

        self.kontrol = tk.Label(text="For Car Tracking :'a'" + "    " + "For Pedestrian Tracking :'i'")
        self.kontrol.place(relx=0, rely=0, relwidth=1)
        self.kontrol.config(font=("Times New Roman", 12, "bold"))
        self.kontrol.config(fg="black")

        self.stream = tk.Button(text="Start Tracking", command=self.thread)
        self.stream.place(relx=0.125, rely=0.25, relwidth=0.75)
        self.stream.config(font=("Times New Roman", 12, "bold"))
        self.stream.config(fg="blue")

        self.turn_back = tk.Button(text="Turn Home Autonomously", command=self.turn)
        self.turn_back.place(relx=0.125, rely=0.5, relwidth=0.75)
        self.turn_back.config(font=("Times New Roman", 12, "bold"))
        self.turn_back.config(fg="blue")

    def turn(self):
        drone.turn_back()

    def havalan(self):
        drone.takeoff()
        time.sleep(5)
        drone.move_up(150)

    def connection(self):
        drone.connect()
        drone.streamon()

    def thread(self):
        videothread = Thread(target=Detector.video_detect(self))
        videothread.run()


arayuz = pencere()
arayuz.title("İHA-OTONOM-TAKİP")
arayuz.geometry("320x240+200+50")
arayuz.mainloop()
