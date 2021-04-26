import torch


def save_checkpoint(epoch, model, optimizer, filename):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    torch.save(state, filename)