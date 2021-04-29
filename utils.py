import torch
import copy


def save_checkpoint(epoch, model, optimizer, filename):
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer}
    torch.save(state, filename)

def save_state_dict(model_path, sd_filename):
    checkpoint = torch.load(model_path)
    net = checkpoint['model']
    sd = net.state_dict()
    new_sd = copy.deepcopy(sd)
    for key in sd.keys():
        new_key = key.replace('network.', '')
        if new_key[0] == '0':
            new_key = 'conv1' + new_key[1:]
        if new_key[0] == '1':
            new_key = 'bn1' + new_key[1:]
        if new_key[0] == '2':
            new_key = 'relu' + new_key[1:]
        if new_key[0] == '3':
            new_key = 'maxpool' + new_key[1:]
        if new_key[0] == '4':
            new_key = 'layer1' + new_key[1:]
        if new_key[0] == '5':
            new_key = 'layer2' + new_key[1:]
        if new_key[0] == '6':
            new_key = 'layer3' + new_key[1:]
        if new_key[0] == '7':
            new_key = 'layer4' + new_key[1:]
        new_sd[new_key] = new_sd.pop(key)
    torch.save(new_sd, filename)

