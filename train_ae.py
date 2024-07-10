from models import AttU_Net, DeepfakeDetector, CNN
import torchfunc
from data_loader_pictures import DeepFakeDataset
import torchvision.models as models
import torch
from torchvision.transforms import transforms
torchfunc.cuda.reset()
import numpy as np
cuda = True if torch.cuda.is_available() else False
import os
import time
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
from double_unet import Double_UNet


model = Double_UNet(output=2)
loss = torch.nn.MSELoss().to('cuda')

dataset_adr = r'E:\full_frames_ff' # r'E:\saved_img'
train_file_path = r'train_test_split.xlsx'
img_type = 'fullface'
dataset = 'FF++'

transf = transforms.Compose([
    transforms.ToTensor(),
])
lr = 1e-3
train_batch_size=16
model_param_adr=r"D:\saved_model\AE_unet_double_d2_l1_fullface_epoch_2_param_all_282_172.pkl"



data_train = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                             batch_size=train_batch_size, train=True, image_type=img_type, dataset=dataset)
if model_param_adr:
    model.load_state_dict(torch.load(model_param_adr))

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
model.cuda()
for epoch in range(0, 5):
    losses = []
    data_train.shuffle()
    for i in range(len(data_train) // 4):

        t, _ = data_train[i]
        t = t.cuda()

        x = model(t)
        l = loss(x, t)

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        losses.append(l.cpu().detach())

        if i%10==0:
            print('Loss', np.mean(losses))

    print(f'\n\n\n\n Epoch {epoch + 1}')
    # Saving model
    torch.save(model.state_dict(),
               os.path.join(r'D:\saved_model',
                            'new_AE__' + img_type + '_epoch_' + str(epoch) + '_param_' + dataset + '_' +
                            str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(
                                time.gmtime()[4]) + '.pkl'))