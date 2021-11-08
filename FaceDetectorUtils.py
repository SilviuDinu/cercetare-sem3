import cv2
import numpy as np
from facenet_pytorch import MTCNN as MTCNN_PYTORCH, InceptionResnetV1
from mtcnn import MTCNN as MTCNN_NORMAL
import os
import sys
import math
import matplotlib.pyplot as plt
from PIL import Image



class FaceDetector(object):
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn

    def _draw(self, frame, boxes, probs, landmarks):

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
                    prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Draw landmarks
                cv2.circle(frame, tuple(ld[0]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[1]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[2]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[3]), 5, (0, 0, 255), -1)
                cv2.circle(frame, tuple(ld[4]), 5, (0, 0, 255), -1)
        except:
            pass

        return frame

    def runVideoDetection(self):
        print('live video detection started')
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            try:
                # detect face box, probability and landmarks
                boxes, probs, landmarks = self.mtcnn.detect(
                    frame, landmarks=True)
                # draw on frame
                self._draw(frame, boxes, probs, landmarks)

            except:
                pass

            # Show the frame
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
                        detector = MTCNN_NORMAL()
                        faces = detector.detect_faces(pixels)
                        # plot each box
                        # self.draw_facebox(filename, faces)
                        try:
                            x, y, width, height = faces[0]['box']
                            crop_img = pixels[y:y+height, x:x+width]
                            if not os.path.exists(os.path.join(outputPath, emotionCategory)):
                                os.makedirs(os.path.join(outputPath, emotionCategory))
                            cv2.imwrite(os.path.join(outputPath, emotionCategory, emotionCategory + '_' + file), crop_img)   
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

                        resnet = InceptionResnetV1(pretrained='vggface2').eval()
                        img = cv2.imread(filename)

                        mtcnn = MTCNN_PYTORCH()
                        # Get cropped and prewhitened image tensor
                        if not os.path.exists(os.path.join(outputPath, emotionCategory)):
                            os.makedirs(os.path.join(outputPath, emotionCategory))
                        img_cropped = mtcnn(img, save_path=os.path.join(outputPath, emotionCategory, emotionCategory + '_' + file))
                            
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
            rect = plt.Rectangle((x, y), width, height, fill=False, color='orange')
            # draw the box
            ax.add_patch(rect)
        # show the plot
        plt.show()




