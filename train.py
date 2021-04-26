from numpy import argmax
from numpy import vstack
from sklearn.metrics import accuracy_score
from utils import save_checkpoint
import logging


def train_model(model, criterion, optimizer, start_epoch, epochs, train_dataset, train_dl, device, filename, log_filename):
    logger = logging.getLogger('part')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.debug('Num training images: {}'.format(len(train_dataset)))
    logger.debug('\nStart training for %d epochs.\n' % epochs)
    print('Num training images: {}'.format(len(train_dataset)))
    print('\nStart training for %d epochs.\n' % epochs)
    for epoch in range(start_epoch, epochs):
        model.train()
        predictions, actuals = list(), list()
        train_loss = 0.0
        for i, batch in enumerate(train_dl):
            part = batch['part'].to(device)
            targets = batch['part_index'].to(device)
            optimizer.zero_grad()
            yhat = model(part)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            train_loss += yhat.shape[0] * loss.item()
            yhat = yhat.cpu().detach().numpy()
            actual = targets.cpu().numpy()
            yhat = argmax(yhat, axis = 1)
            actual = actual.reshape((len(actual), 1))
            yhat = yhat.reshape((len(yhat), 1))
            predictions.append(yhat)
            actuals.append(actual)
        predictions, actuals = vstack(predictions), vstack(actuals)
        train_acc = accuracy_score(actuals, predictions)
        train_loss /= len(train_dataset)
        logger.debug('epoch %d: train_loss: %f, train_acc: %f' % (epoch + 1, train_loss, train_acc))
        print('epoch %d: train_loss: %f, train_acc: %f' % (epoch + 1, train_loss, train_acc))
        save_checkpoint(epoch, model, optimizer, filename)        