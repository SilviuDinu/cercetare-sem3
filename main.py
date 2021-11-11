import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import os
import sys
from utils import *
from PIL import Image


curr = os.path.dirname(__file__)
emotionsDict = {
    "Angry": 0,
    "Disgust": 1,
    "Fear": 2,
    "Happy": 3,
    "Neutral": 4,
    "Sad": 5,
    "Surprise": 6,
}

def main():

    if not os.path.exists(os.path.join(curr, r'fer2013_resized/train/')):
        input_path = os.path.join(curr, r'fer2013/train/')
        output_path = os.path.join(curr, r'fer2013_resized/train/')
        resizeImages(input_path, output_path)
       # os.system('afplay /System/Library/Sounds/Glass.aiff')



    if not os.path.exists(os.path.join(curr, r'fer2013_resized/validation/')):
        input_path = os.path.join(curr, r'fer2013/validation/')
        output_path = os.path.join(curr, r'fer2013_resized/validation/')
        resizeImages(input_path, output_path)
      #  os.system('afplay /System/Library/Sounds/Glass.aiff')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # preffered image size by resnet is (224, 224, 3)
    # the below also include the normalize transform from above
    data_transforms = {
        'train':
            transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]),
        'validation':
            transforms.Compose([
                # transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])}

    image_datasets = {
        'train':
            datasets.ImageFolder(os.path.join(curr, r'fer2013_resized/train/'), data_transforms['train']),
        'validation':
            datasets.ImageFolder(os.path.join(curr, r'fer2013_resized/validation/'), data_transforms['validation'])}

    dataloaders = {
        'train':
            torch.utils.data.DataLoader(
                image_datasets['train'],
                batch_size=32,
                shuffle=True,
                num_workers=4),
        'validation':
            torch.utils.data.DataLoader(
                image_datasets['validation'],
                batch_size=32,
                shuffle=False,
                num_workers=4)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model = models.resnet50(pretrained=True).to(device)

    for param in model.parameters():
        # change this to false!!!!!!!!!!!!!!!!!!!
        # This freezes the inner layers
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 7)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters())

    def train_model(model, criterion, optimizer, num_epochs=3):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)
        #    os.system('say Starting epoch ' + str(epoch+1))

            for phase in ['train', 'validation']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        # print('phase train if statement')

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # sys.stdout.flush()
                    # sys.stdout.write("\bRunning loss: %s. Running corrects: %s %%\r" %
                    #                  (running_loss, running_corrects))

                epoch_loss = running_loss / len(image_datasets[phase])
                epoch_acc = running_corrects.double() / \
                    len(image_datasets[phase])

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
        return model
    
    model_trained = train_model(model, criterion, optimizer, num_epochs=32)

    if not os.path.exists(os.path.join(curr, r'models/')):
        os.makedirs(os.path.join(curr, r'models/'))
    torch.save(model_trained.state_dict(),
               os.path.join(curr, r'models/mobilenetv2_32epochs_F_nll_loss.h5'))

   # os.system('afplay /System/Library/Sounds/Glass.aiff')


    model = models.resnet50(pretrained=False).to(device)
    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 7)).to(device)
    model.load_state_dict(torch.load(os.path.join(curr, r'models/mobilenetv2_32epochs_F_nll_loss.h5')))

    validation_img_paths = []

    for root, dirs, files in os.walk(os.path.join(curr, r'fer2013/validation/')):
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
                    if (len(validation_img_paths) < 7):
                        validation_img_paths.append(filename)

            except IOError:
                pass

    img_list = [Image.open(img_path) for img_path in validation_img_paths]

    validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                    for img in img_list])

    pred_logits_tensor = model(validation_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        ax.set_title("{:.0f}% Angry, {:.0f}% Disgust, {:.0f}% Fear, {:.0f}% Happy, {:.0f}% Neutral, {:.0f}% Sad, {:.0f}% Surprise,".format(100*pred_probs[i, 0],
                                                                                                                                           100*pred_probs[i, 2],
                                                                                                                                           100*pred_probs[i, 3],
                                                                                                                                           100*pred_probs[i, 4],
                                                                                                                                           100*pred_probs[i, 5],
                                                                                                                                           100*pred_probs[i, 6],
                                                                                                                                           100*pred_probs[i, 7]))
        ax.imshow(img)


if __name__ == "__main__":
    main()
    #os.system('afplay /System/Library/Sounds/Glass.aiff')


