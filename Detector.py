import time

import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(200)

class Detector:
    def __init__(self):
        pass

    def readClasses(self, classFilePath):
        with open(classFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
            #ColorsList
            self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))
            # print(len(self.classesList), len(self.colorList))

    def downloadModel(self, model_URL):
        filename = os.path.basename(model_URL)
        self.modelName = filename.split('.')[0]

        self.cache_dir = "./pretrained_Models"
        if(os.path.exists(self.cache_dir)):
            pass
        else:
            os.makedirs(self.cache_dir)

        get_file(fname=filename, origin=model_URL, cache_dir=self.cache_dir,
                 cache_subdir="datasets", extract=True)


    def loadModel(self):
        print("Loading the model:", self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cache_dir, "datasets", self.modelName, "saved_model"))
        print("Model " + self.modelName + " loaded successfully")

    def createBoundingBox(self, image, threshold=0.5):
        # converting Image to Tensor
        image_numpy = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_numpy, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.model(input_tensor) # Returns Dictionary
        # print("Keys:",detections.keys())
        bboxes = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imgH, imgW, imgC = image.shape
        bboxIdx = tf.image.non_max_suppression(bboxes, classScores, max_output_size=50,
                                               iou_threshold=threshold, score_threshold=threshold)

        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxes[i].tolist())

                classConfidence = round(100*classScores[i])
                classIndex = classIndexes[i]

                classLabelText = self.classesList[classIndex].upper()
                classColor = self.colorList[classIndex]

                displayText = '{}, {}%'.format(classLabelText, classConfidence)

                # setting detections w.r.t the image size
                ymin, xmin, ymax, xmax = bbox
                ymin, xmin, ymax, xmax = (ymin * imgH), (xmin * imgW), (ymax * imgH), (xmax * imgW)
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)

                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=2)
                cv2.putText(image, displayText, (xmax, ymax), cv2.FONT_HERSHEY_PLAIN, 1, classColor, 2)

        return image

    def predict_Img(self, imgPath, threshold=0.5):
        image = cv2.imread(imgPath)
        bboxImage = self.createBoundingBox(image, threshold)

        cv2.imwrite(self.modelName + ".jpg", bboxImage)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print("Showing The image")

    def predict_Video(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(videoPath)

        if (cap.isOpened() == False):
            print("Error opening file...")
            return

        (success, image) = cap.read()
        start_time = 0

        while success:
            current_time = time.time()

            fps = 1/(current_time - start_time)
            start_time = current_time

            bboxImage = self.createBoundingBox(image, threshold)
            cv2.putText(bboxImage, "FPS: "+str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 2)
            cv2.imshow("Result ", bboxImage)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            (success, image) = cap.read()

        cv2.destroyAllWindows()















