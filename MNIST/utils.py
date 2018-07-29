from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
import torch.nn.init as init
import copy

# custom weights initialization


def weights_init(m):
    classname = m.__class__.__name__
    """
    # Conv系全てに対しての初期化
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    """
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, nonlinearity='relu')  # initialize Conv
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # initialize for BN
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        # initialize Linear
        init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        m.bias.data.normal_(0, 0.02)


def train_model(model, dataloader, criterion, optimizer, num_epochs, device, scheduler=None):
    since = datetime.datetime.now()
    epochs = tqdm(range(num_epochs), desc="Epoch", unit='epoch')
    phases = ['train', 'val']
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # train loop
    for epoch in epochs:
        for phase in phases:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            train_loss = 0.0
            train_acc = 0
            # Iterate over data.
            iteration = tqdm(dataloader[phase],
                             desc="{} iteration".format(phase.capitalize()),
                             unit='iter')
            for inputs, labels in iteration:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    # returns loss is mean_wise
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # statistics
                train_loss += loss.item() * inputs.size(0)
                train_acc += torch.sum(preds == labels.data)
            epoch_loss = train_loss / len(inputs)
            epoch_acc = train_acc.double() / len(inputs)
            tqdm.write('{} Epoch: {} Loss: {:.4f} Acc: {:.4f}'.format(
                epoch, phase.capitalize(), epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        tqdm.write("")

    time_elapsed = datetime.datetime.now() - since
    tqdm.write('Training complete in {}'.format(time_elapsed))
    tqdm.write('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
