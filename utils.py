import torch
from FaceDetectorUtils import FaceDetector
# from facenet_pytorch import MTCNN, InceptionResnetV1
# from mtcnn import MTCNN
import numpy as np
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageFilter
import sys
import math
import cv2

curr = os.path.dirname(__file__)
fcd = FaceDetector()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

emotionsDict = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Surprise": 6,
}


def processFacesFromPictures():
    fcd.runPictureDetection_MTCNN_PYTORCH(os.path.join(
        curr, r'fer2013/train/'), os.path.join(curr, r'processed/pytorch_MTCNN/train/'))
    fcd.runPictureDetection_MTCNN_PYTORCH(os.path.join(
        curr, r'fer2013/validation/'), os.path.join(curr, r'processed/pytorch_MTCNN/validation/'))
   # os.system('afplay /System/Library/Sounds/Glass.aiff')


def startVideoCapture():
    fcd.runVideoDetection()




def centralizeImages(outputPath):
    if not outputPath:
        outputPath = os.path.join(curr, r'processed/centralized/')
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    trainingInputPath = os.path.join(curr, r'fer2013/train/')
    moveImagesTo(trainingInputPath, os.path.join(outputPath, r'train/'))
    testInputPath = os.path.join(curr, r'fer2013/validation/')
    moveImagesTo(testInputPath, os.path.join(outputPath, r'test/'))


def moveImagesTo(input, output):
    if not os.path.exists(output):
        os.makedirs(output)
        for root, dirs, files in os.walk(input):
            for idx, file in enumerate(files):
                pathList = os.path.normpath(root).split(os.path.sep)
                emotionCategory = pathList[len(pathList) - 1]
                filename = os.path.join(input, emotionCategory, file)
                # filename = emotionCategory + '_' + file
                try:
                    if not file.startswith('.'):
                        sys.stdout.flush()
                        sys.stdout.write("\bCurrent progress: %s %%\r" %
                                         (str(math.ceil(idx / len(files) * 100))))

                        img = cv2.imread(filename)
                        cv2.imwrite(os.path.join(
                            output, emotionCategory + '_' + file), img)
                except IOError:
                    pass

def resizeImages(inputPath, outputPath):
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
                    image = Image.open(filename)
                    img = image.resize((224, 224), Image.ANTIALIAS) 
                    sharpened = img.filter(ImageFilter.SHARPEN)

                    if not os.path.exists(os.path.join(outputPath, emotionCategory)):
                        os.makedirs(os.path.join(outputPath, emotionCategory))
                    
                    img.save(os.path.join(outputPath, emotionCategory, file), quality=95)

            except IOError:
                print("cannot create thumbnail for '%s'" % file)
    sys.stdout.write("\b\nDone\n")


class TrainingData(torch.utils.data.Dataset):
    def __init__(self):
        inputPath = os.path.join(curr, r'processed/centralized/train')
        xy = []
        for root, dirs, files in os.walk(inputPath):
            for idx, file in enumerate(files):
                if not file.startswith('.'):
                    sys.stdout.flush()
                    sys.stdout.write("\bCurrent progress: %s %%\r" %
                                     (str(math.ceil(idx / len(files) * 100))))

                    pathList = os.path.normpath(root).split(os.path.sep)
                    # emotionCategory = pathList[len(pathList) - 1]
                    emotionCategory = file.split('_')[0]
                    filename = os.path.join(inputPath, file)
                    img = Image.open(filename).convert('L')
                    # data = list(img.getdata())
                    # width, height = img.size
                    # pixels = [pixels[i * width:(i + 1) * width] for i in xrange(height)]
                    img_as_tensor = transforms.ToTensor()(img)
                    # img = np.array(Image.open(filename).convert('RGB'))
                    # img = transforms.functional.to_tensor(img)
                    # img = transforms.functional.resize(
                    #     img, (48, 48), interpolation=Image.BILINEAR)
                    pixels_numpy = img_as_tensor.numpy()
                    # if idx < 1:
                    #   print(pixels_numpy,squeeze(0), pixels_numpy.squeeze(0).shape)
                    # print(img_as_tensor, type(
                    #     img_as_tensor), img_as_tensor.shape)
                    # output = [emotionsDict[emotionCategory]]
                    # print(type(image), type(output))
                    # xy = img_as_tensor.numpy()
                    # np.append(pixels_numpy, float(emotionsDict[emotionCategory]))
                    output = torch.tensor([emotionsDict[emotionCategory]])
                    # tensor = torch.stack((img_as_tensor, output), 0)
                    # print(img_as_tensor, torch.tensor(float(emotionsDict[emotionCategory])))
                    xy.append([img_as_tensor, output])
                    # print(xy[0])
                    # print(type(pairs))
                    # print('XY', np.array(xy), np.array(xy).shape)

        xy = np.array(xy, dtype=dtype)
        print(xy[0])
        self.x = torch.from_numpy(xy[:, 0:-1])
        self.y = torch.from_numpy(xy[:, [-1]])
        self.n_samples = xy.shape[0]
        # print(xy[0], xy[1])
        print('========================')
        print(self.x, self.x.shape, type(self.x))
        print('========================')
        print(self.y, self.y.shape, type(self.y))
        print(self.n_samples)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


#processFacesFromPictures()