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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from prettytable import PrettyTable

curr = os.path.dirname(__file__)

emotionsDict_3 = {
    0: "Angry",
    1: "Happy",
    2: "Sad"
}


def main():

    if not os.path.exists(os.path.join(curr, r'fer2013_resized/train/')):
        input_path = os.path.join(curr, r'fer2013_copy/train/')
        output_path = os.path.join(curr, r'fer2013_resized/train/')
        resizeImages(input_path, output_path)

    if not os.path.exists(os.path.join(curr, r'fer2013_resized/validation/')):
        input_path = os.path.join(curr, r'fer2013_copy/validation/')
        output_path = os.path.join(curr, r'fer2013_resized/validation/')
        resizeImages(input_path, output_path)

    data_transforms = {
        'train':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
        'validation':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])}

    image_datasets = {
        'train':
            datasets.ImageFolder(os.path.join(
                curr, r'fer2013_copy/train/'), data_transforms['train']),
        'validation':
            datasets.ImageFolder(os.path.join(curr, r'fer2013_copy/validation/'), data_transforms['validation'])}

    dataloaders = {
        'train':
            torch.utils.data.DataLoader(
                image_datasets['train'],
                batch_size=10,
                shuffle=True,
                num_workers=2),
        'validation':
            torch.utils.data.DataLoader(
                image_datasets['validation'],
                batch_size=10,
                shuffle=False,
                num_workers=2)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model_mobilenetv2 = models.mobilenet_v2(pretrained=True).to(device)
    model_resnet50 = models.resnet50(pretrained=True).to(device)
    model_vgg16 = models.vgg16(pretrained=True).to(device)

    # for param in model.parameters():
    #     param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    optimizer_mobilenetv2 = optim.Adam(model_mobilenetv2.parameters())
    optimizer_resnet50 = optim.Adam(model_resnet50.parameters())
    optimizer_vgg16 = optim.Adam(model_vgg16.parameters())

    def train_model(model, criterion, optimizer, num_epochs=3):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train', 'validation']:
                predictions_1 = []
                predictions_2 = []
                predictions_3 = []
                targets = []
                correct_1 = 0
                correct_2 = 0
                correct_3 = 0
                total_1 = 0
                total_2 = 0
                total_3 = 0

                if phase == 'train':
                    model.train()

                else:
                    model.eval()

                running_loss_1 = 0.0
                running_loss_2 = 0.0
                running_loss_3 = 0.0
                running_corrects_1 = 0
                running_corrects_2 = 0
                running_corrects_3 = 0

                for inputs, labels in dataloaders[phase]:
                    torch.cuda.empty_cache()

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs_1 = model(inputs)
                    # outputs_2 = models[1](inputs)
                    # outputs_3 = models[2](inputs)

                    loss_1 = criterion(outputs_1, labels)
                    # loss_2 = criterion(outputs_2, labels)
                    # loss_3 = criterion(outputs_3, labels)

                    # for idx, i in enumerate(outputs_1):
                    #     predictions_1.append(torch.argmax(i).long())
                    #     targets.append(labels[idx].squeeze().long())
                    #     if torch.argmax(i) == labels[idx]:
                    #         correct_1 += 1
                    #     total_1 += 1

                    # for idx, i in enumerate(outputs_2):
                    #     predictions_2.append(torch.argmax(i).long())
                    #     targets.append(labels[idx].squeeze().long())
                    #     if torch.argmax(i) == labels[idx]:
                    #         correct_2 += 1
                    #     total_2 += 1

                    # for idx, i in enumerate(outputs_3):
                    #     predictions_3.append(torch.argmax(i).long())
                    #     targets.append(labels[idx].squeeze().long())
                    #     if torch.argmax(i) == labels[idx]:
                    #         correct_3 += 1
                    #     total_3 += 1

                    if phase == 'train':
                        # for optimizer in optimizers:
                        optimizer.zero_grad()
                        loss_1.backward()
                        # loss_2.backward()
                        # loss_3.backward()
                        # for optimizer in optimizers:
                        optimizer.zero_grad()

                    _1, preds_1 = torch.max(outputs_1, 1)
                    running_loss_1 += loss_1.item() * inputs.size(0)
                    running_corrects_1 += torch.sum(preds_1 == labels.data)
                    # _2, preds_2 = torch.max(outputs_2, 1)
                    # running_loss_2 += loss_2.item() * inputs.size(0)
                    # running_corrects_2 += torch.sum(preds_2 == labels.data)
                    # _3, preds_3 = torch.max(outputs_3, 1)
                    # running_loss_3 += loss_3.item() * inputs.size(0)
                    # running_corrects_3 += torch.sum(preds_3 == labels.data)

                epoch_loss_1 = running_loss_1 / len(image_datasets[phase])
                epoch_acc_1 = running_corrects_1.double() / \
                    len(image_datasets[phase])

                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss_1,
                                                            epoch_acc_1))

                # epoch_loss_2 = running_loss_2 / len(image_datasets[phase])
                # epoch_acc_2 = running_corrects_2.double() / \
                #     len(image_datasets[phase])

                # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                #                                             epoch_loss_2,
                #                                             epoch_acc_2))

                # epoch_loss_3 = running_loss_3 / len(image_datasets[phase])
                # epoch_acc_3 = running_corrects_3.double() / \
                #     len(image_datasets[phase])

                # print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                #                                             epoch_loss_3,
                #                                             epoch_acc_3))
        return model

    models_trained = train_model(
        model_mobilenetv2, criterion, optimizer_mobilenetv2, num_epochs=1)

    # torch.save(model_trained.state_dict(),
    #            os.path.join(curr, r'models/mobilenet_5_02_2022.h5'))

    print(models_trained, type(models_trained))

    # nb_classes = 3

    # model = models.mobilenet_v2(pretrained=False).to(device)

    # model.load_state_dict(torch.load(os.path.join(
    #     curr, r'models/mobilenet_5_02_2022.h5'), map_location=torch.device(device)))
    # model.eval()

    # with torch.no_grad():
    #     for phase in ['train', 'validation']:
    #         predictions = []
    #         targets = []
    #         correct = 0
    #         total = 0
    #         running_loss = 0.0
    #         running_corrects = 0
    #         for inputs, labels in dataloaders[phase]:
    #             inputs = inputs.to(device)
    #             labels = labels.to(device)

    #             outputs = model(inputs)
    #             loss = criterion(outputs, labels)

    #             _, preds = torch.max(outputs, 1)
    #             running_loss += loss.item() * inputs.size(0)
    #             running_corrects += torch.sum(preds == labels.data)

    #             epoch_loss = running_loss / len(image_datasets[phase])
    #             epoch_acc = running_corrects.double() / \
    #                 len(image_datasets[phase])

    #             for idx, i in enumerate(outputs):
    #                 predictions.append(torch.argmax(i).long())
    #                 targets.append(labels[idx].squeeze().long())
    #                 if torch.argmax(i) == labels[idx]:
    #                     correct += 1
    #                 total += 1
    #         print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
    #                                                     epoch_loss,
    #                                                     epoch_acc))

    #         accuracy = round(correct/total, 3)
    #         print("Accuracy: ", accuracy)

    #         predictions = torch.tensor(predictions)
    #         targets = torch.tensor(targets)
    #         conf_matrix = confusion_matrix(predictions, targets)
    #         conf_matrix = torch.tensor(conf_matrix)

    #         t = PrettyTable(['Class', 'Name', 'TP', 'TN', 'FP', 'FN',
    #                         'Sensitivity', 'Specificity', 'Precision', 'F1'])

    #         TP = np.diag(conf_matrix)
    #         for c in range(nb_classes):
    #             idx = torch.ones(nb_classes).bool()
    #             idx[c] = 0

    #             # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
    #             TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()

    #             FP = conf_matrix[idx, c].sum()

    #             FN = conf_matrix[c, idx].sum()

    #             sensitivity = (TP[c] / (TP[c]+FN))
    #             specificity = (TN / (TN+FP))
    #             precision = (TP[c] / (TP[c] + FP))
    #             F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

    #             t.add_row([c, emotionsDict_3[c], float(TP[c]), float(TN), float(FP), float(FN), float(
    #                 sensitivity), round(float(specificity), 3), float(precision), round(float(F1), 3)])

    #         print(t)

    # validation_img_paths = []

    # for root, dirs, files in os.walk(os.path.join(curr, r'Testing/')):
    #     for idx, file in enumerate(files):
    #         pathList = os.path.normpath(root).split(os.path.sep)
    #         emotionCategory = pathList[len(pathList) - 1]
    #         filename = os.path.join('Testing/', emotionCategory, file)
    #         filename_out = os.path.join('Testing/', emotionCategory, file)
    #         img = Image.open(filename)
    #         rgbimg = img.convert('RGB')
    #         rgbimg.save(filename_out)
    #         # filename = emotionCategory + '_' + file
    #         try:
    #             if not file.startswith('.'):
    #                 sys.stdout.flush()
    #                 sys.stdout.write("\bCurrent progress: %s %%\r" %
    #                                  (str(math.ceil(idx / len(files) * 100))))
    #                 if (len(validation_img_paths) < 7):
    #                     validation_img_paths.append(filename_out)

    #         except IOError:
    #             pass

    # validation_img_paths = [
    #     'Testing/1.jpg',
    #     'Testing/4.jpg',
    #     'Testing/8.jpg',
    #     'Testing/10.jpg'
    # ]

    # img_list = [Image.open(img_path) for img_path in validation_img_paths]

    # validation_batch = torch.stack([data_transforms['validation'](img).to(device)
    #                                 for img in img_list])

    # pred_logits_tensor = model(validation_batch)
    # pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    # print(img_list)
    # fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    # for i, img in enumerate(img_list):
    #     ax = axs[i]
    #     ax.axis('off')
    #     ax.set_title("{:.0f}% Angry, \n{:.0f}% Happy, \n{:.0f}% Sad ".format(100*pred_probs[i, 0],
    #                                                                          100 *
    #                                                                          pred_probs[i, 1],
    #                                                                          100*pred_probs[i, 2]))
    #     ax.imshow(img)

    # plt.show()

    # img_list = []

    # validation_img_paths = [
    #     'Testing/19.jpg',
    #     'Testing/30.jpg',
    #     'Testing/eu2.jpg'
    # ]
    # validation_batch = []

    # img_list = [Image.open(img_path) for img_path in validation_img_paths]

    # validation_batch = torch.stack([data_transforms['validation'](img).to(device)
    #                                 for img in img_list])

    # pred_logits_tensor = model(validation_batch)
    # pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    # print(img_list)
    # fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    # for i, img in enumerate(img_list):
    #     ax = axs[i]
    #     ax.axis('off')
    #     ax.set_title("{:.0f}% Angry, \n{:.0f}% Happy, \n{:.0f}% Sad ".format(100*pred_probs[i, 0],
    #                                                                          100 *
    #                                                                          pred_probs[i, 1],
    #                                                                          100*pred_probs[i, 2]))
    #     ax.imshow(img)

    # plt.show()


if __name__ == "__main__":
    main()
