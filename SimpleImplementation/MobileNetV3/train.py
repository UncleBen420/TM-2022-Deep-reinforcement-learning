import matplotlib.pyplot as plt
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from model import build_model
from dataset import get_datasets
from utils import save_model, save_plots

# Training function.
def train(
        model, trainloader, optimizer,
        criterion, scheduler=None, epoch=None
):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    iters = len(trainloader)
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        torch.cuda.empty_cache()
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the weights.
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch + i / iters)

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion, class_names):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    # We need two lists to keep track of class-wise accuracy.
    class_correct = list(0. for i in range(len(class_names)))
    class_total = list(0. for i in range(len(class_names)))
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            # Calculate the accuracy for each class.
            correct = (preds == labels).squeeze()
            for i in range(len(preds)):
                label = labels[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = (valid_running_correct / len(testloader.dataset))
    # Print the accuracy for each class after every epoch.
    print('\n')
    for i in range(len(class_names)):
        print(f"Accuracy of class {class_names[i]}: {class_correct[i] / class_total[i]}")
    print('\n')
    return epoch_loss, epoch_acc


if __name__ == '__main__':

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    dataset_classes = ['ship', 'not-ship']

    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epochs', type=int, default=10,
        help='Number of epochs to train our network for'
    )
    parser.add_argument(
        '-lr', '--learning-rate', type=float,
        dest='learning_rate', default=0.001,
        help='Learning rate for training the model'
    )
    parser.add_argument(
        '-pw', '--pretrained', action='store_true',
        help='whether to use pretrained weihgts or not'
    )
    parser.add_argument(
        '-ft', '--fine-tune', dest='fine_tune', action='store_true',
        help='whether to train all layers or not'
    )

    parser.add_argument(
        '-d', '--data', help='where the dataset is'
    )

    args = vars(parser.parse_args())

    # Load the training and validation datasets.
    train_loader, valid_loader = get_datasets(args['data'])

    # Learning_parameters.
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    # Load the model.
    model = build_model(
        pretrained=args['pretrained'],
        fine_tune=args['fine_tune'],
        num_classes=len(dataset_classes)
    ).to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=1,
        verbose=True
    )
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(
            model, train_loader,
            optimizer, criterion,
            scheduler=scheduler, epoch=epoch
        )
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                     criterion, dataset_classes)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)
        time.sleep(5)

    plt.plot(train_loss, label="train loss")
    plt.plot(valid_loss, label="test loss")
    plt.plot(train_acc, label="train acc")
    plt.plot(valid_acc, label="test acc")

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion)
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    print('TRAINING COMPLETE')