# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import imutils
from imutils.video import VideoStream

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')



def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def run(input, model_dir, device_id, confirence):
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    vs = VideoStream(src=input).start()

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, height=480, width=640)        # TODO change regard to hardware capabilities
        image = imutils.rotate_bound(frame, 90)                     # TODO: I am using webcam so I rotated my webcam and input image to get 3/4 aspect ratio

        result = check_image(image)
        if result is False:
            break

        image_bbox = model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model_test.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time()-start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label]/2
        if label == 1:
            if value < confirence:
                print(f'Low confirence - {value}')
                continue

            print("Image is Real Face. Score: {:.2f}.".format(value))
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (0, 255, 0)
        else:
            print("Image is Fake Face. Score: {:.2f}.".format(value))
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

        cv2.imshow("Result", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        required=False,
        help="image used to test")
    parser.add_argument(
        '--input',
        type=int,
        required=False,
        default=0,
        help="camera device id"
    )
    parser.add_argument(
        '--confirence',
        required=False,
        type=float,
        default=0.9,
        help="minimum confirence for liveness detection"
    )
    args = parser.parse_args()
    run(args.input, args.model_dir, args.device_id, args.confirence)