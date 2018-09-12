import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.interpolate

VIDEOS_DIR = 'video'

def graph(x, y, title, xlabel='', ylabel=''):
        plt.figure(num=title)
        plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.margins(0)


class ObjectTracker:

    def __init__(self, file, start_time=0, end_time=float('inf')):
        self.file = file
        self.start_time = start_time
        self.end_time = end_time

    def process_video(self, show=False):
        cap = cv2.VideoCapture(self.file)
        start_time = time.time()

        times = []
        positions = []

        while cap.isOpened():
            if cv2.waitKey(25) & 0xFF == 'q':
                break

            ret, frame = cap.read()
            if ret is False:
                break

            img, cur_pos = self.process_img(frame)

            
            if show:
                cv2.imshow(self.file, img)

            rows, cols, _ = img.shape

            cur_time = time.time() - start_time

            if cur_time > self.end_time:
                break

            if cur_pos is not None and cur_time > self.start_time:
                cur_pos = self.pixels_to_meters(cur_pos)
                times.append(cur_time)
                positions.append(cur_pos)
                print("Time, {}  Position: {}".format(cur_time, cur_pos))

        if show:
            cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()
        return times, positions

    def get_new_pos_vel_and_accel(self, times, positions, k=2):
        pos_spl = scipy.interpolate.UnivariateSpline(x=times, y=positions, k=k, s=1e8)
        vel_spl = pos_spl.derivative()
        acc_spl = vel_spl.derivative()

        x_new = np.linspace(self.start_time, self.end_time, num=1000)
        return pos_spl(times), vel_spl(times), acc_spl(times), times

    def process_img(self, img):
        pass

    def pixels_to_meters(self, pixels):
        pass


class TennisBallTracker(ObjectTracker):

    def __init__(self, file):
        super().__init__(file, start_time=4.8, end_time=8.2)
        self.lower_hsv_range = np.array([28.0, 89.0, 45.0])
        self.upper_hsv_range = np.array([154.0, 187.0, 170.0])
        self.min_radius = 8.0

    def process_img(self, img):
        rows, cols, _ = img.shape
        img_resized = img[int(0.42*rows):int(0.67*rows), int(0.04*cols):]

        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(img_hsv, self.lower_hsv_range, self.upper_hsv_range)
        mask = cv2.erode(mask, None, iterations=4)
        mask = cv2.dilate(mask, None, iterations=4)

        cv2.imshow('mask', mask)

        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x = None
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > self.min_radius:
                cv2.circle(img_resized, (int(x), int(y)), int(radius), (0, 255, 255), 2)    
            else:
                x = None

        return img_resized, x

    def pixels_to_meters(self, pixels):
        return pixels * 2.1 / 500

class BaseBallTracker(ObjectTracker):

    def __init__(self, file):
        super().__init__(file, start_time=12.5, end_time=18.3)
        self.lower_hsv_range = np.array([13.0, 64.0, 19.0])
        self.upper_hsv_range = np.array([122.0, 120.0, 255.0])
        self.min_radius = 8.0

    def process_img(self, img):
        rows, cols, _ = img.shape
        img_resized = img[int(0.5*rows):int(0.67*rows), int(0.04*cols):]

        img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(img_hsv, self.lower_hsv_range, self.upper_hsv_range)
        mask = cv2.erode(mask, None, iterations=4)
        mask = cv2.dilate(mask, None, iterations=4)

        cv2.imshow('mask', mask)

        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        x = None
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > self.min_radius:
                cv2.circle(img_resized, (int(x), int(y)), int(radius), (0, 255, 255), 2)    
            else:
                x = None

        return img_resized, x

    def pixels_to_meters(self, pixels):
        return pixels * 2.2 / 500


def calculate_and_show(tracker):
    times, positions = tracker.process_video(show=True)
    pos, vel, accel, times = tracker.get_new_pos_vel_and_accel(times, positions)
    graph(times, pos, 'position', xlabel='Time (s)', ylabel='Position (m)')
    graph(times, vel, 'velocity', xlabel='Time (s)', ylabel='Velocity (m/s)')
    graph(times, accel, 'acceleration', xlabel='Time (s)', ylabel='Acceleration (m/s^2)')
    plt.show()

def main():
    # tennisball_tracker = TennisBallTracker('video/tennisball.MOV')
    # calculate_and_show(tennisball_tracker)

    baseball_tracker = BaseBallTracker('video/tennisball_and_baseball.MOV')
    calculate_and_show(baseball_tracker)

if __name__ == '__main__':
    main()