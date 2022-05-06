import cv2
import glob
import os
from datetime import datetime


def video_to_frames(path):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    # 帧率
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # 总帧数
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))

    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        cv2.imwrite("frames/%d.png" % (i), frame)
    return
def sort_test():
    a = ['abc',1]
    ab = ['abc',0,12]
    abb = ['abc',122]
    abbb = ['abc',10]
    print(sorted([a,ab,abb,abbb]))

if __name__ == '__main__':
    # t1 = datetime.now()
    # video_to_frames("IMG_7180.MOV")
    # t2 = datetime.now()
    # print("Time cost = ", (t2 - t1))
    # print("SUCCEED !!!")
    sort_test()
