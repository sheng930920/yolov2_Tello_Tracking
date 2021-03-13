import time
import cv2
import numpy as np
from djitellopy import Tello
from darkflow.net.build import TFNet

drone = Tello()
drone.connect()
drone.streamon()
options = {"model": "cfg/yolov2-custom-ozan.cfg",#custom config for task
           "load": "ckpt/yolov2-custom-ozan_best70.weights",#custom weights for task
           "gpu": 1.0,
           "threshold": 0.4
           }# determine configs

yolo_net = TFNet(options)
#drone.takeoff()
drone.get_battery()

class Detector:

    def boxing(self, originalimage, predictions):
        global right_left_velocity, forw_back_velocity, up_down_velocity, yaw_velocity
        newImage = np.copy(originalimage)
        for result in predictions:
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']
            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']
            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3))
            target = result['label']

            if confidence > 0.4 and target == 'insan':# check confidence value and target class for tracking
                newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255, 0, 0), 3)#draw boundary box
                newImage = cv2.putText(newImage, label, (top_x, top_y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                                       (0, 230, 0), 1, cv2.LINE_AA)#labeling
                target_x = top_x + (btm_x - top_x) / 2
                target_y = top_y + (btm_y - top_y) / 2
                cv2.arrowedLine(newImage, (480, 320), (int(target_x), int(target_y)), (0, 255, 255), 2)
        return newImage

    def video_detect(self):
        while 1:
            start_time = time.time()
            image = drone.get_frame_read().frame
            original_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = yolo_net.return_predict(original_img)
            image = self.boxing(image, results)
            cv2.imshow('Image', image)
            FPS = 1.0 / (time.time() - start_time)
            print(f'FPS={FPS}')
            
            fps_log = open('fps_log_@416x416.txt', 'a+')#save FPS logs
            fps_log.write(str(FPS) + '\n')
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                process_time = drone.get_flight_time()
                print(process_time)
                drone.end()
                break
        cv2.destroyAllWindows()

def main():
    detector = Detector()
    detector.video_detect()

if __name__ == '__main__':
    main()
