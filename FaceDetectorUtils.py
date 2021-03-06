import cv2
import numpy as np
from facenet_pytorch import MTCNN as MTCNN_PYTORCH, InceptionResnetV1
from mtcnn import MTCNN as MTCNN_NORMAL
import os
import torch
import sys
import math
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim


data_transforms = {
    'train':
        transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
    'validation':
        transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])}

emotionsDict = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Surprise": 6,
}

emotionsDict_3 = {
    0: "Angry",
    1: "Happy",
    2: "Sad"
}

emotionsDict_7 = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise"
}


class FaceDetector():
    def __init__(self):
        self.mtcnn_normal = MTCNN_NORMAL()
        self.mtcnn_pytorch = MTCNN_PYTORCH()

    def _draw(self, frame, boxes, probs, landmarks, text):

        # Draw landmarks and boxes for each face detected

        try:
            for box, prob, ld in zip(boxes, probs, landmarks):
                # Draw rectangle on frame
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255),
                              thickness=2)

                # Show probability
                cv2.putText(frame, str(
                    text), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw landmarks
                # cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                # cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except:
            pass

        return frame

    def runVideoDetection(self):
        print('live video detection started')
        cap = cv2.VideoCapture(0)
        model = models.mobilenet_v2(pretrained=True)
        model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3))
        model.load_state_dict(torch.load(
            'models/Random_Affine_Horizontal_Flip_76_48_acc_mobilenetv2_Angry_Sad_Happy.h5', map_location='cpu'))
        model.eval()
        while True:
            ret, frame = cap.read()
            try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn_pytorch.detect(
                    frame, landmarks=True)

                # draw on frame
                # self._draw(frame, boxes, probs, landmarks)

            except:
                pass

            # Show the frame
            mtcnn = self.mtcnn_pytorch
            img_cropped = mtcnn(frame, save_path='frame.jpg')
            img = cv2.imread('frame.jpg')
            resized = cv2.resize(img, (48, 48))
            cv2.imwrite('frame.jpg', resized)
            good = Image.open('frame.jpg')
            tensorImg = transforms.ToTensor()(good)

            # pred_logits_tensor = model(tensorImg.unsqueeze(0))
            output = model(tensorImg.unsqueeze(0))
            pred_probs = F.softmax(output, dim=1).cpu().data.numpy()
            pred = torch.argmax(output)
            # print(100*pred_probs[0, int(pred)])
            frame_legend = '{}, {:.2f}%'.format(
                emotionsDict_3[int(pred)], 100*pred_probs[0, int(pred)])
            
            self._draw(frame, boxes, probs, landmarks, frame_legend)
            cv2.imshow('Face Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def runPictureDetection_MTCNN_NORMAL(self, inputPath, outputPath):
        for root, dirs, files in os.walk(inputPath):
            for idx, file in enumerate(files):
                pathList = os.path.normpath(root).split(os.path.sep)
                emotionCategory = pathList[len(pathList) - 1]
                filename = os.path.join(inputPath, emotionCategory, file)
                try:
                    if not file.startswith('.'):
                        sys.stdout.flush()
                        sys.stdout.write("\bCurrent progress: %s %%\r" %
                                         (str(math.ceil(idx / len(files) * 100))))

                        pixels = cv2.imread(filename)
                        # detect faces in the image
                        detector = self.mtcnn_normal
                        faces = detector.detect_faces(pixels)
                        # plot each box
                        # self.draw_facebox(filename, faces)
                        try:
                            x, y, width, height = faces[0]['box']
                            crop_img = pixels[y:y+height, x:x+width]
                            if not os.path.exists(os.path.join(outputPath, emotionCategory)):
                                os.makedirs(os.path.join(
                                    outputPath, emotionCategory))
                            cv2.imwrite(os.path.join(
                                outputPath, emotionCategory, emotionCategory + '_' + file), crop_img)
                        except IndexError:
                            pass
                            # create the shape

                except IOError:
                    print("cannot create thumbnail for '%s'" % file)
        sys.stdout.write("\b\nDone\n")

    def runPictureDetection_MTCNN_PYTORCH(self, inputPath, outputPath):
        for root, dirs, files in os.walk(inputPath):
            for idx, file in enumerate(files):
                pathList = os.path.normpath(root).split(os.path.sep)
                emotionCategory = pathList[len(pathList) - 1]
                filename = os.path.join(inputPath, emotionCategory, file)
                try:
                    if not file.startswith('.'):
                        sys.stdout.flush()
                        sys.stdout.write("\bCurrent progress: %s %%\r" %
                                         (str(math.ceil(idx / len(files) * 100))))

                        resnet = InceptionResnetV1(
                            pretrained='vggface2').eval()
                        img = cv2.imread(filename)

                        mtcnn = self.mtcnn_pytorch
                        # Get cropped and prewhitened image tensor
                        if not os.path.exists(os.path.join(outputPath, emotionCategory)):
                            os.makedirs(os.path.join(
                                outputPath, emotionCategory))
                        img_cropped = mtcnn(img, save_path=os.path.join(
                            outputPath, emotionCategory, emotionCategory + '_' + file))

                except IOError:
                    print("cannot create thumbnail for '%s'" % file)
        sys.stdout.write("\b\nDone\n")

    def draw_facebox(self, filename, result_list):
        # load the image
        data = plt.imread(filename)
        # plot the image
        plt.imshow(data)
        # get the context for drawing boxes
        ax = plt.gca()
        # plot each box
        for result in result_list:
            # get coordinates
            x, y, width, height = result['box']
            # create the shape
            rect = plt.Rectangle((x, y), width, height,
                                 fill=False, color='orange')
            # draw the box
            ax.add_patch(rect)
        # show the plot
        plt.show()
