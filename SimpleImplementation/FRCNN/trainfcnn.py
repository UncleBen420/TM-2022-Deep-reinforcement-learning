import argparse
import os
import random

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from DOTADatasetLoader import DOTA
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    #transforms.append(T.NormalizeMobileNet())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes, weights_path=""):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if weights_path != "":
        model.load_state_dict(torch.load(weights_path))

    return model


def train(path_train, path_test, path_saved_model, weights, num_epochs):
    # use our dataset and defined transformations
    dataset = DOTA(path_train, get_transform(train=True))
    dataset_test = DOTA(path_test, get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and boat
    num_classes = 2

    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes, weights)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    history = []
    for epoch in range(num_epochs):

        # train for one epoch, printing every 10 iterations
        losses = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        print(losses.items())
        for key, value in losses.items():
            losses[key] = value.avg

        history.append(losses)
        print(losses)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), os.path.join(path_saved_model,"weights"))

    history = {key: [i[key] for i in history] for key in history[0]}

    return model, history


def predict(image, model, detection_threshold, device="cpu"):
    """
    Predict the output of an image after forward pass through
    the model and return the bounding boxes, class names, and
    class labels.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # mobilenet use a special normalisation
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # transform the image to tensor
    image = transform(image).to(device)
    # add a batch dimension
    image = image.unsqueeze(0)
    # get the predictions on the image
    with torch.no_grad():
        outputs = model(image)
    # get score for all the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # get all the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes


def draw_boxes(boxes, image):
    """
    Draws the bounding box around a detected object.
    """
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = [255, 0, 0]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
    return image


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='This program allow user to train a faster RCNN mobilenetv3 on a dataset')
    parser.add_argument('-tr', '--train_path',
                        help='the path to the data. it must contains a images and a labels folder')
    parser.add_argument('-ts', '--test_path',
                        help='the path to the data. it must contains a images and a labels folder')
    parser.add_argument('-o', '--saved_model_path',
                        help='the path where the model will be saved')
    parser.add_argument('-w', '--weights', help='the path to the weights if needed')
    parser.add_argument('-e', '--epochs', help=' number of epochs')

    args = parser.parse_args()

    if args.weights is None:
        args.weights = ""

    model, history = train(args.train_path, args.test_path, args.saved_model_path, args.weights, int(args.epochs))
    test_image = random.choice([x for x in os.listdir(os.path.join(args.test_path,'images'))])
    test_image = Image.open(os.path.join(args.test_path,'images', test_image)).convert("RGB")

    bboxes = predict(test_image, model, 0.5)
    test_image = draw_boxes(bboxes, test_image)
    print(bboxes)

    for key, loss in history.items:
        plt.plot(loss, label=key)
    plt.show()

    plt.imshow(test_image)
    plt.show()
