# pyinstaller -F --hidden-import openpyxl grab.py
# pyinstaller==6.1.0
# pyinstaller-hooks-contrib==2023.10
# opencv-python==4.8.1.78
# mss==9.0.1
# numpy==1.23.4
# pandas==1.3.5
# Flask-Cors==3.0.10
# Flask==2.2.3
# Werkzeug==2.2.3
import cv2
import mss
import numpy as np
# from pandas import read_excel
from flask_cors import CORS
from PIL import ImageFile
from flask import Flask, Response
import argparse


ImageFile.LOAD_TRUNCATED_IMAGES = True
# excel_data = read_excel("./屏幕信息.xlsx").iloc[0, :]
# Moniter = excel_data["屏幕"]
# x = excel_data["起始x"]
# y = excel_data["起始y"]
# width = excel_data["宽度"]
# height = excel_data["高度"]

class LoadScreenshots:
    def __init__(self, source, transforms=None):
        source, *params = source.split()
        self.screen, left, top, width, height = 0, None, None, None, None
        if len(params) == 1:
            self.screen = int(params[0])
        elif len(params) == 4:
            left, top, width, height = (int(x) for x in params)
        elif len(params) == 5:
            self.screen, left, top, width, height = (int(x) for x in params)

        self.transforms = transforms
        self.mode = 'stream'
        self.frame = 0
        self.sct = mss.mss()

        monitor = self.sct.monitors[self.screen]
        self.top = monitor["top"] if top is None else (monitor["top"] + top)
        self.left = monitor["left"] if left is None else (monitor["left"] + left)
        self.width = width or monitor["width"]
        self.height = height or monitor["height"]
        self.monitor = {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def __iter__(self):
        return self

    def __next__(self):
        im0 = np.array(self.sct.grab(self.monitor))[:, :, :3]
        # s = f"screen {self.screen} (LTWH): {self.left},{self.top},{self.width},{self.height}: "

        if self.transforms:
            im0 = self.transforms(im0)
        # else:
        #     im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]
        #     im = im.transpose((2, 0, 1))[::-1]
        #     im = np.ascontiguousarray(im)
        # self.frame += 1
        # return str(self.screen), im, im0, None, s
        return im0

def gen(dataloader):
    """视频流生成"""
    dataset = dataloader
    # for path, im, im0s, vid_cap, s in dataset:
    for im0 in dataset:
        # im0s = cv2.resize(im0s,(1920,1080))
        data = cv2.imencode('.jpg', im0)[1]
        frame = data.tobytes()
        yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#开始主程序
app = Flask(__name__)

CORS(app, supports_credentials=True)

@app.route('/video_feed')
def video_feed():
    global Moniter, x, y, width, height
    return Response(gen(LoadScreenshots(f"Screen {Moniter} {x} {y} {width} {height}")), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    # with open("./屏幕信息.txt", "r") as f:
    #     ip, port = f.readline().split()
    #     port = int(port)
    #     Moniter, x, y, width, height = f.readline().split()
    #     Moniter = int(Moniter)
    #     x = int(x)
    #     y = int(y)
    #     width = int(width)
    #     height = int(height)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, default='0.0.0.0', help="ip address")
    parser.add_argument("--port", type=int, default=5000, help="port number")
    parser.add_argument("--Moniter", type=int, default=0, help="Moniter number")
    parser.add_argument("--x", type=int, default=0, help="x")
    parser.add_argument("--y", type=int, default=0, help="y")
    parser.add_argument("--width", type=int, default=1920, help="width")
    parser.add_argument("--height", type=int, default=1080, help="height")
    args = parser.parse_args()

    print(f"屏幕{args.Moniter}的起始坐标为({args.x},{args.y}),宽度为{args.width},高度为{args.height}")
    print(f'http://{args.ip}:{args.port}')
    app.run(host=args.ip, port=args.port, debug=False, threaded=True, processes=True)
