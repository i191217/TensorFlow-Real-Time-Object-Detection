from Detector import *

classFile = 'coco.names'
threshold=0.5
img_path = "pretrained_Models/images/federer.jpg"
model_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu" \
            "-8.tar.gz "
# model_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d4_coco17_tpu-32.tar.gz"

video_path = "pretrained_Models/images/person-bicycle-car-detection.mp4"
# video_path = 0 (for Webcam)
# video_path = 1 (for External Webcam I guess...)

detector = Detector()

detector.readClasses(classFile)
detector.downloadModel(model_URL)
detector.loadModel()

detector.predict_Img(img_path, threshold)
detector.predict_Video(video_path, threshold)

