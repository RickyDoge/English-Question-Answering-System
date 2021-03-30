import torch
import torch.nn as nn
from model.baseline import BaselineModel
from model.intensive_reading_ca import IntensiveReadingWithCrossAttention
from model.intensive_reading_cnn import IntensiveReadingWithConvolutionNet

# a short script to convert multi-gpu model to single-gpu/cpu model

hidden_dim = 768
which_model = 'google/electra-base-discriminator'

#model = BaselineModel(clm_model=which_model, hidden_dim=hidden_dim)
model = IntensiveReadingWithCrossAttention(clm_model=which_model, hidden_dim=hidden_dim)
#model = IntensiveReadingWithConvolutionNet(clm_model=which_model, hidden_dim=hidden_dim, out_channel=100)

if torch.cuda.device_count() > 1:
    device = torch.cuda.current_device()
    model.to(device)
    model = nn.DataParallel(module=model)
    print('Use Multi GPUs. Number of GPUs: ', torch.cuda.device_count())
elif torch.cuda.device_count() == 1:
    device = torch.cuda.current_device()
    model.to(device)
    print('Use 1 GPU')
else:
    device = torch.device('cpu')  # CPU
    print("use CPU")

model.load_state_dict(torch.load('model_parameters.pth'))
torch.save(model.module.state_dict(), '../single_gpu_model.pth')
