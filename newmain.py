import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import os
from torch.autograd import Variable
import sys
from utils import *
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from prettytable import PrettyTable
import torch.nn.functional as F


curr = os.path.dirname(__file__)

emotionsDict_3 = {
    0: "Angry",
    1: "Happy",
    2: "Sad"
}


def main():

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
                batch_size=16,
                shuffle=True,
                num_workers=4),
        'validation':
            torch.utils.data.DataLoader(
                image_datasets['validation'],
                batch_size=16,
                shuffle=False,
                num_workers=4)}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model_mobilenet_v2 = models.mobilenet_v2(pretrained=True).to(device)
    model_resnet50 = models.resnet50(pretrained=True).to(device)
    model_mobilenet_v3_large = models.mobilenet_v3_large(
        pretrained=True).to(device)
    # model_vgg16 = models.vgg16(pretrained=True).to(device)

    for param in model_resnet50.parameters():
        param.requires_grad = False

    model_resnet50.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, 3)).to(device)

    # model_inception_v3.AuxLogits.fc = nn.Linear(768, 3)
    # model_inception_v3.fc = nn.Linear(2048, 3)
    # model_vgg16.classifier[6] = nn.Linear(4096, 3).to(device)

    model_mobilenet_v2.classifier[1] = nn.Linear(1280, 3).to(device)
    model_mobilenet_v3_large.classifier[3] = nn.Linear(1280, 3).to(device)

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    optimizer_mobilenetv2 = optim.Adam(model_mobilenet_v2.parameters())
    optimizer_resnet50 = optim.Adam(model_resnet50.parameters())
    optimizer_model_mobilenetv3_large = optim.Adam(
        model_mobilenet_v3_large.parameters())
    # optimizer_vgg16 = optim.Adam(model_vgg16.parameters())
    # optimizer_inceptionv3 = optim.Adam(model_inception_v3.parameters())

    def train_model(models, criterion, optimizers, num_epochs=3):
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

            for phase in ['train']:
                if phase == 'train':
                    models[0].train()
                    models[1].train()
                    models[2].train()
                else:
                    models[0].eval()
                    models[1].eval()
                    models[2].eval()

                running_loss_1 = 0.0
                running_corrects_1 = 0
                running_loss_2 = 0.0
                running_corrects_2 = 0
                running_loss_3 = 0.0
                running_corrects_3 = 0
                running_loss_global = 0.0
                running_corrects_global = 0

                for inputs, labels in dataloaders[phase]:
                    # torch.cuda.empty_cache()
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs_1 = models[0](inputs)
                    outputs_2 = models[1](inputs)
                    outputs_3 = models[2](inputs)

                    correct_outputs = torch.clone(outputs_1)

                    for idx, i in enumerate(correct_outputs):
                        claims = []
                        pred_mobilenet = torch.argmax(outputs_1[idx])
                        pred_resnet = torch.argmax(outputs_2[idx])
                        pred_mobilenetv3 = torch.argmax(outputs_3[idx])
                        claims.append(pred_mobilenet)
                        claims.append(pred_resnet)
                        claims.append(pred_mobilenetv3)

                        most_frequent = max(set(claims), key=claims.count)
                        if most_frequent == pred_mobilenet:
                            correct_outputs[idx] = outputs_1[idx]
                        elif most_frequent == pred_resnet:
                            correct_outputs[idx] = outputs_2[idx]
                        elif most_frequent == pred_mobilenetv3:
                            correct_outputs[idx] = outputs_3[idx]

                    loss_1 = criterion(outputs_1, labels)
                    loss_2 = criterion(outputs_2, labels)
                    loss_3 = criterion(outputs_3, labels)
                    loss_4 = criterion(correct_outputs, labels)

                    if phase == 'train':
                        optimizers[0].zero_grad()
                        optimizers[1].zero_grad()
                        optimizers[2].zero_grad()
                        loss_1.backward()
                        loss_2.backward()
                        loss_3.backward()
                        optimizers[0].step()
                        optimizers[1].step()
                        optimizers[2].step()

                    _1, preds_1 = torch.max(outputs_1, 1)
                    running_loss_1 += loss_1.item() * inputs.size(0)
                    running_corrects_1 += torch.sum(preds_1 == labels.data)
                    _2, preds_2 = torch.max(outputs_2, 1)
                    running_loss_2 += loss_2.item() * inputs.size(0)
                    running_corrects_2 += torch.sum(preds_2 == labels.data)
                    _3, preds_3 = torch.max(outputs_3, 1)
                    running_loss_3 += loss_3.item() * inputs.size(0)
                    running_corrects_3 += torch.sum(preds_3 == labels.data)
                    _4, preds_4 = torch.max(correct_outputs, 1)
                    running_loss_global += loss_4.item() * inputs.size(0)
                    running_corrects_global += torch.sum(
                        preds_4 == labels.data)

                    loss_1 = 0
                    loss_2 = 0
                    loss_3 = 0
                    loss_4 = 0
                    inputs = []
                    labels = []

                epoch_loss_1 = running_loss_1 / len(image_datasets[phase])
                epoch_acc_1 = running_corrects_1.double() / \
                    len(image_datasets[phase])

                print('-' * 10)
                print('MobilenetV2: {} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                                         epoch_loss_1,
                                                                         epoch_acc_1))
                epoch_loss_2 = running_loss_2 / len(image_datasets[phase])
                epoch_acc_2 = running_corrects_2.double() / \
                    len(image_datasets[phase])

                print('ResNet50: {} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                                      epoch_loss_2,
                                                                      epoch_acc_2))

                epoch_loss_3 = running_loss_3 / len(image_datasets[phase])
                epoch_acc_3 = running_corrects_3.double() / \
                    len(image_datasets[phase])

                print('MobilenetV3: {} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                                         epoch_loss_3,
                                                                         epoch_acc_3))

                epoch_loss_global = running_loss_global / \
                    len(image_datasets[phase])
                epoch_acc_global = running_corrects_global.double() / \
                    len(image_datasets[phase])

                print('Global results: {} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                                            epoch_loss_global,
                                                                            epoch_acc_global))
                print('-' * 10)
        return [
            models[0],
            models[1],
            models[2]
        ]

    # models_trained = train_model([model_mobilenet_v2, model_resnet50, model_mobilenet_v3_large], criterion, [
    #                              optimizer_mobilenetv2, optimizer_resnet50, optimizer_model_mobilenetv3_large], num_epochs=10)

    # torch.save(models_trained[0].state_dict(),
    #            os.path.join(curr, r'models/mobilenetv2_8-epochs_voting_3.h5'))
    # torch.save(models_trained[1].state_dict(),
    #            os.path.join(curr, r'models/resnet50_8-epochs_voting_3.h5'))
    # torch.save(models_trained[2].state_dict(),
    #            os.path.join(curr, r'models/mobilenetv3_8-epochs_voting_2.h5'))

    nb_classes = 3

    model_mobilenet_v2 = models.mobilenet_v2(pretrained=False).to(device)
    model_resnet50 = models.resnet50(pretrained=False).to(device)
    model_mobilenet_v3_large = models.mobilenet_v3_large(
        pretrained=True).to(device)

    model_resnet50.fc = nn.Sequential(
        nn.Linear(2048, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(128, 3)).to(device)

    model_mobilenet_v2.classifier[1] = nn.Linear(1280, 3).to(device)
    model_mobilenet_v3_large.classifier[3] = nn.Linear(1280, 3).to(device)

    model_mobilenet_v2.load_state_dict(torch.load(os.path.join(
        curr, r'models/mobilenetv2_8-epochs_voting_2.h5'), map_location=torch.device(device)))
    model_mobilenet_v2.eval()
    model_resnet50.load_state_dict(torch.load(os.path.join(
        curr, r'models/resnet50_8-epochs_voting_2.h5'), map_location=torch.device(device)))
    model_resnet50.eval()
    model_mobilenet_v3_large.load_state_dict(torch.load(os.path.join(
        curr, r'models/mobilenetv3_8-epochs_voting.h5'), map_location=torch.device(device)))
    model_mobilenet_v3_large.eval()

    with torch.no_grad():
        for phase in ['validation']:
            predictions_1 = []
            predictions_2 = []
            predictions_3 = []
            predictions_global = []
            targets = []
            correct = 0
            total = 0
            running_loss_1 = 0.0
            running_corrects_1 = 0
            running_loss_2 = 0.0
            running_corrects_2 = 0
            running_loss_3 = 0.0
            running_corrects_3 = 0
            running_loss_global = 0.0
            running_corrects_global = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs_1 = model_mobilenet_v2(inputs)
                outputs_2 = model_resnet50(inputs)
                outputs_3 = model_mobilenet_v3_large(inputs)

                correct_outputs = torch.clone(outputs_1)

                for idx, i in enumerate(correct_outputs):
                    claims = []
                    pred_mobilenet = torch.argmax(outputs_1[idx])
                    pred_resnet = torch.argmax(outputs_2[idx])
                    pred_inceptionv3 = torch.argmax(outputs_3[idx])
                    claims.append(pred_mobilenet)
                    claims.append(pred_resnet)
                    claims.append(pred_inceptionv3)

                    most_frequent = max(set(claims), key=claims.count)
                    if most_frequent == pred_mobilenet:
                        correct_outputs[idx] = outputs_1[idx]
                    elif most_frequent == pred_resnet:
                        correct_outputs[idx] = outputs_2[idx]
                    elif most_frequent == pred_inceptionv3:
                        correct_outputs[idx] = outputs_3[idx]

                loss_1 = criterion(outputs_1, labels)
                loss_2 = criterion(outputs_2, labels)
                loss_3 = criterion(outputs_3, labels)
                loss_4 = criterion(correct_outputs, labels)

                _1, preds_1 = torch.max(outputs_1, 1)
                running_loss_1 += loss_1.item() * inputs.size(0)
                running_corrects_1 += torch.sum(preds_1 == labels.data)
                _2, preds_2 = torch.max(outputs_2, 1)
                running_loss_2 += loss_2.item() * inputs.size(0)
                running_corrects_2 += torch.sum(preds_2 == labels.data)
                _3, preds_3 = torch.max(outputs_3, 1)
                running_loss_3 += loss_3.item() * inputs.size(0)
                running_corrects_3 += torch.sum(preds_3 == labels.data)
                _4, preds_4 = torch.max(correct_outputs, 1)
                running_loss_global += loss_4.item() * inputs.size(0)
                running_corrects_global += torch.sum(
                    preds_4 == labels.data)

                epoch_loss_1 = running_loss_1 / \
                    len(image_datasets[phase])
                epoch_acc_1 = running_corrects_1.double() / \
                    len(image_datasets[phase])

                epoch_loss_2 = running_loss_2 / \
                    len(image_datasets[phase])
                epoch_acc_2 = running_corrects_2.double() / \
                    len(image_datasets[phase])

                epoch_loss_3 = running_loss_3 / \
                    len(image_datasets[phase])
                epoch_acc_3 = running_corrects_3.double() / \
                    len(image_datasets[phase])

                epoch_loss_global = running_loss_global / \
                    len(image_datasets[phase])
                epoch_acc_global = running_corrects_global.double() / \
                    len(image_datasets[phase])

                for idx, i in enumerate(outputs_1):
                    predictions_1.append(torch.argmax(i).long())
                    targets.append(labels[idx].squeeze().long())
                for idx, i in enumerate(outputs_2):
                    predictions_2.append(torch.argmax(i).long())
                for idx, i in enumerate(outputs_3):
                    predictions_3.append(torch.argmax(i).long())
                for idx, i in enumerate(correct_outputs):
                    predictions_global.append(torch.argmax(i).long())

            print('-' * 10)
            print('Mobilenetv2 {} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                                    epoch_loss_1,
                                                                    epoch_acc_1))
            print('Resnet50 {} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                                 epoch_loss_2,
                                                                 epoch_acc_2))
            print('MobilenetV3 {} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                                    epoch_loss_3,
                                                                    epoch_acc_3))
            print('Global {} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                               epoch_loss_global,
                                                               epoch_acc_global))
            print('-' * 10)
            predictions_1 = torch.tensor(predictions_1)
            predictions_2 = torch.tensor(predictions_2)
            predictions_3 = torch.tensor(predictions_3)
            predictions_global = torch.tensor(predictions_global)
            targets = torch.tensor(targets)
            conf_matrix_1 = confusion_matrix(predictions_1, targets)
            conf_matrix_2 = confusion_matrix(predictions_2, targets)
            conf_matrix_3 = confusion_matrix(predictions_3, targets)
            conf_matrix_global = confusion_matrix(predictions_global, targets)
            conf_matrix_1 = torch.tensor(conf_matrix_1)
            conf_matrix_2 = torch.tensor(conf_matrix_2)
            conf_matrix_3 = torch.tensor(conf_matrix_3)
            conf_matrix_global = torch.tensor(conf_matrix_global)

            t_1 = PrettyTable(['Class', 'Name', 'TP', 'TN', 'FP', 'FN',
                               'Sensitivity', 'Specificity', 'Precision', 'F1'])

            TP = np.diag(conf_matrix_1)
            for c in range(nb_classes):
                idx = torch.ones(nb_classes).bool()
                idx[c] = 0

                # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
                TN = conf_matrix_1[idx.nonzero()[:, None], idx.nonzero()].sum()

                FP = conf_matrix_1[idx, c].sum()

                FN = conf_matrix_1[c, idx].sum()

                sensitivity = (TP[c] / (TP[c]+FN))
                specificity = (TN / (TN+FP))
                precision = (TP[c] / (TP[c] + FP))
                F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

                t_1.add_row([c, emotionsDict_3[c], float(TP[c]), float(TN), float(FP), float(FN), float(
                    sensitivity), round(float(specificity), 3), float(precision), round(float(F1), 3)])

            print(t_1)

            t_2 = PrettyTable(['Class', 'Name', 'TP', 'TN', 'FP', 'FN',
                               'Sensitivity', 'Specificity', 'Precision', 'F1'])

            TP = np.diag(conf_matrix_2)
            for c in range(nb_classes):
                idx = torch.ones(nb_classes).bool()
                idx[c] = 0

                # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
                TN = conf_matrix_2[idx.nonzero()[:, None], idx.nonzero()].sum()

                FP = conf_matrix_2[idx, c].sum()

                FN = conf_matrix_2[c, idx].sum()

                sensitivity = (TP[c] / (TP[c]+FN))
                specificity = (TN / (TN+FP))
                precision = (TP[c] / (TP[c] + FP))
                F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

                t_2.add_row([c, emotionsDict_3[c], float(TP[c]), float(TN), float(FP), float(FN), float(
                    sensitivity), round(float(specificity), 3), float(precision), round(float(F1), 3)])

            print(t_2)

            t_3 = PrettyTable(['Class', 'Name', 'TP', 'TN', 'FP', 'FN',
                               'Sensitivity', 'Specificity', 'Precision', 'F1'])

            TP = np.diag(conf_matrix_3)
            for c in range(nb_classes):
                idx = torch.ones(nb_classes).bool()
                idx[c] = 0

                # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
                TN = conf_matrix_3[idx.nonzero()[:, None], idx.nonzero()].sum()

                FP = conf_matrix_3[idx, c].sum()

                FN = conf_matrix_3[c, idx].sum()

                sensitivity = (TP[c] / (TP[c]+FN))
                specificity = (TN / (TN+FP))
                precision = (TP[c] / (TP[c] + FP))
                F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

                t_3.add_row([c, emotionsDict_3[c], float(TP[c]), float(TN), float(FP), float(FN), float(
                    sensitivity), round(float(specificity), 3), float(precision), round(float(F1), 3)])

            print(t_3)

            t_global = PrettyTable(['Class', 'Name', 'TP', 'TN', 'FP', 'FN',
                                    'Sensitivity', 'Specificity', 'Precision', 'F1'])

            TP = np.diag(conf_matrix_global)
            for c in range(nb_classes):
                idx = torch.ones(nb_classes).bool()
                idx[c] = 0

                # conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
                TN = conf_matrix_global[idx.nonzero()[:, None], idx.nonzero()].sum()

                FP = conf_matrix_global[idx, c].sum()

                FN = conf_matrix_global[c, idx].sum()

                sensitivity = (TP[c] / (TP[c]+FN))
                specificity = (TN / (TN+FP))
                precision = (TP[c] / (TP[c] + FP))
                F1 = 2 * (precision * sensitivity) / (precision + sensitivity)

                t_global.add_row([c, emotionsDict_3[c], float(TP[c]), float(TN), float(FP), float(FN), float(
                    sensitivity), round(float(specificity), 3), float(precision), round(float(F1), 3)])

            print(t_global)

    validation_img_paths = [
        'Testing/1.jpg',
        'Testing/4.jpg',
        'Testing/8.jpg',
        'Testing/10.jpg'
    ]

    img_list = [Image.open(img_path) for img_path in validation_img_paths]

    validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                    for img in img_list])

    pred_logits_tensor_1 = model_mobilenet_v2(validation_batch)
    pred_logits_tensor_2 = model_resnet50(validation_batch)
    pred_logits_tensor_3 = model_mobilenet_v3_large(validation_batch)

    pred_logits_tensor_global = torch.clone(pred_logits_tensor_1)

    for idx, i in enumerate(pred_logits_tensor_global):
        claims = []
        pred_mobilenet = torch.argmax(pred_logits_tensor_1[idx])
        pred_resnet = torch.argmax(pred_logits_tensor_2[idx])
        pred_mobilenetv3 = torch.argmax(pred_logits_tensor_3[idx])
        claims.append(pred_mobilenet)
        claims.append(pred_resnet)
        claims.append(pred_mobilenetv3)

        most_frequent = max(set(claims), key=claims.count)
        if most_frequent == pred_mobilenet:
            pred_logits_tensor_global[idx] = pred_logits_tensor_1[idx]
        elif most_frequent == pred_resnet:
            pred_logits_tensor_global[idx] = pred_logits_tensor_2[idx]
        elif most_frequent == pred_mobilenetv3:
            pred_logits_tensor_global[idx] = pred_logits_tensor_3[idx]

    pred_probs_1 = F.softmax(pred_logits_tensor_1, dim=1).cpu().data. numpy()
    pred_probs_2 = F.softmax(pred_logits_tensor_2, dim=1).cpu().data. numpy()
    pred_probs_3 = F.softmax(pred_logits_tensor_3, dim=1).cpu().data. numpy()
    pred_probs_global = F.softmax(
        pred_logits_tensor_global, dim=1).cpu().data. numpy()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        fig.suptitle('MobilenetV2', fontsize=16)
        ax.set_title("{:.0f}% Angry, \n{:.0f}% Happy, \n{:.0f}% Sad ".format(100*pred_probs_1[i, 0],
                                                                             100 *
                                                                             pred_probs_1[i, 1],
                                                                             100*pred_probs_1[i, 2]))
        ax.imshow(img)

    plt.show()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        fig.suptitle('Resnet50', fontsize=16)
        ax.set_title("{:.0f}% Angry, \n{:.0f}% Happy, \n{:.0f}% Sad ".format(100*pred_probs_2[i, 0],
                                                                             100 *
                                                                             pred_probs_2[i, 1],
                                                                             100*pred_probs_2[i, 2]))
        ax.imshow(img)

    plt.show()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        fig.suptitle('Mobilenetv3', fontsize=16)
        ax.set_title("{:.0f}% Angry, \n{:.0f}% Happy, \n{:.0f}% Sad ".format(100*pred_probs_3[i, 0],
                                                                             100 *
                                                                             pred_probs_3[i, 1],
                                                                             100*pred_probs_3[i, 2]))
        ax.imshow(img)

    plt.show()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        fig.suptitle('Global', fontsize=16)
        ax.set_title("Global {:.0f}% Angry, \n{:.0f}% Happy, \n{:.0f}% Sad ".format(100*pred_probs_global[i, 0],
                                                                                    100 *
                                                                                    pred_probs_global[i, 1],
                                                                                    100*pred_probs_global[i, 2]))
        ax.imshow(img)

    plt.show()

    img_list = []

    validation_img_paths = [
        'Testing/19.jpg',
        'Testing/30.jpg',
        'Testing/eu2.jpg'
    ]
    validation_batch = []

    img_list = [Image.open(img_path) for img_path in validation_img_paths]

    validation_batch = torch.stack([data_transforms['validation'](img).to(device)
                                    for img in img_list])

    pred_logits_tensor_1 = model_mobilenet_v2(validation_batch)
    pred_logits_tensor_2 = model_resnet50(validation_batch)
    pred_logits_tensor_3 = model_mobilenet_v3_large(validation_batch)

    pred_logits_tensor_global = torch.clone(pred_logits_tensor_1)

    for idx, i in enumerate(pred_logits_tensor_global):
        claims = []
        pred_mobilenet = torch.argmax(pred_logits_tensor_1[idx])
        pred_resnet = torch.argmax(pred_logits_tensor_2[idx])
        pred_mobilenetv3 = torch.argmax(pred_logits_tensor_3[idx])
        claims.append(pred_mobilenet)
        claims.append(pred_resnet)
        claims.append(pred_mobilenetv3)

        most_frequent = max(set(claims), key=claims.count)
        if most_frequent == pred_mobilenet:
            pred_logits_tensor_global[idx] = pred_logits_tensor_1[idx]
        elif most_frequent == pred_resnet:
            pred_logits_tensor_global[idx] = pred_logits_tensor_2[idx]
        elif most_frequent == pred_mobilenetv3:
            pred_logits_tensor_global[idx] = pred_logits_tensor_3[idx]

    pred_probs_1 = F.softmax(pred_logits_tensor_1, dim=1).cpu().data. numpy()
    pred_probs_2 = F.softmax(pred_logits_tensor_2, dim=1).cpu().data. numpy()
    pred_probs_3 = F.softmax(pred_logits_tensor_3, dim=1).cpu().data. numpy()
    pred_probs_global = F.softmax(
        pred_logits_tensor_global, dim=1).cpu().data. numpy()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        fig.suptitle('MobilenetV2', fontsize=16)
        ax.set_title("{:.0f}% Angry, \n{:.0f}% Happy, \n{:.0f}% Sad ".format(100*pred_probs_1[i, 0],
                                                                             100 *
                                                                             pred_probs_1[i, 1],
                                                                             100*pred_probs_1[i, 2]))
        ax.imshow(img)

    plt.show()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        fig.suptitle('Resnet50', fontsize=16)
        ax.set_title("{:.0f}% Angry, \n{:.0f}% Happy, \n{:.0f}% Sad ".format(100*pred_probs_2[i, 0],
                                                                             100 *
                                                                             pred_probs_2[i, 1],
                                                                             100*pred_probs_2[i, 2]))
        ax.imshow(img)

    plt.show()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        fig.suptitle('Mobilenetv3', fontsize=16)
        ax.set_title("{:.0f}% Angry, \n{:.0f}% Happy, \n{:.0f}% Sad ".format(100*pred_probs_3[i, 0],
                                                                             100 *
                                                                             pred_probs_3[i, 1],
                                                                             100*pred_probs_3[i, 2]))
        ax.imshow(img)

    plt.show()

    fig, axs = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
        ax = axs[i]
        ax.axis('off')
        fig.suptitle('Global', fontsize=16)
        ax.set_title("{:.0f}% Angry, \n{:.0f}% Happy, \n{:.0f}% Sad ".format(100*pred_probs_global[i, 0],
                                                                             100 *
                                                                             pred_probs_global[i, 1],
                                                                             100*pred_probs_global[i, 2]))
        ax.imshow(img)

    plt.show()


if __name__ == "__main__":
    main()
