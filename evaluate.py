from models import DeepfakeDetector
import torchfunc
from data_loader_pictures_fullvideo import DeepFakeDataset
import numpy as np
import copy
import torch
import pandas as pd
import random
import time
import os
from torchvision import transforms
from utils import AverageMeter
from sklearn.metrics import roc_auc_score


test_setup = pd.DataFrame(columns=['dataset_adr', 'train_file_path', 'dataset'])
test_setup.loc[0] = [r'D:\saved_celefdf_all', r'train_test_celebdf_corect.xlsx', 'celebDF']
test_setup.loc[1] = [r'D:\saved_img', r'test_train_dfdc_final.xlsx', 'dfdc']
test_setup.loc[2] = [r'D:\ff++\saved_images', r'train_test_split.xlsx', 'FF++']


########################################################################################################################
model_param_adr = r"F:\saved_model\deepfake_detector_global_epoch_10002_param_FF++_143_031.pkl" #r'D:\saved_model\Xception_fullface_epoch_77_param_FF++_33_1446.pkl'
images_per_folder = 500
setup_order = [0, 1]
########################################################################################################################
text2write = ''

for setup_nr in setup_order:

    dataset_adr = test_setup.loc[setup_nr, 'dataset_adr']
    train_file_path = test_setup.loc[setup_nr, 'train_file_path']
    dataset = test_setup.loc[setup_nr, 'dataset']

    img_type = 'fullface'
    dataset_model = 'FF++'
    model_type = 'Xception'
    test_batch_size = 1

    print('Model:', model_param_adr)
    print('Setup:', dataset, dataset_adr, train_file_path)

    transf = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_test = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                                batch_size=1, train=False, image_type=img_type, dataset=dataset,
                                images_per_folder=images_per_folder, image_doesnt_contain='compressed', labels_repeat=False)
    data_test.shuffle(random_state=1)

    model = DeepfakeDetector(pretrained=True, finetuning=False, architecture='Xception')
    model.load_state_dict(torch.load(model_param_adr))
    model.cuda()
    print('Starting TEST')

    criterion = torch.nn.BCELoss()

    test_losses = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()
    times = AverageMeter()
    test_predictions_vect = []
    test_targets_vect = []
    data_test.shuffle(random_state=1)
    test_df = copy.copy(data_test.label_df)
    t_start = time.time()

    model.eval()
    for i in range(len(data_test)):

        t1 = time.time()
        print(f'{i}/{len(data_test)}')
        data, targets = data_test[i]
        data, targets = data.cuda(), targets.cuda()

        with torch.no_grad():
            try:
                outputs_gpu = model(data)
                outputs = outputs_gpu.to('cpu').flatten()
            except Exception as e:
                print(e)
                print(f'Failed in test i={i}')
                continue

        test_predictions_vect.append(outputs.mean())
        test_targets_vect.append(targets.to('cpu').detach())
        print(outputs.mean())
        print(targets)

        # test_predictions_vect.append(outputs)
        # targets = targets.to('cpu').detach().repeat(outputs.shape[0])
        # test_targets_vect.append(targets)

        if len(test_predictions_vect) > 1:
            # print(f'Predictions: {test_predictions_vect}')
            # print(f'Targets: {test_targets_vect}')

            try:
                loss = criterion(torch.stack(test_predictions_vect).flatten().type(torch.DoubleTensor),
                                 torch.stack(test_targets_vect).flatten().type(torch.DoubleTensor))
                print(f'Loss: {loss}')
            except Exception as e:
                print(e)

        try:
            if len(torch.unique(torch.cat(test_targets_vect).flatten())) > 1:
                auc_test = roc_auc_score(torch.stack(test_targets_vect).flatten().type(torch.DoubleTensor),
                                         torch.stack(test_predictions_vect).flatten().type(torch.DoubleTensor))
            else:
                auc_test = '-'
        except Exception as e:
            print(e)
            auc_test = 'failed'

        print(f'AUC: {auc_test}')

        t2 = time.time()
        duration_1_sample = t2 - t1
        times.update(duration_1_sample, 1)
        print('Est time:' + str(int(times.avg * (len(data_test) - i) // 3600)) + 'h' +
              str(int(
                  (times.avg * (len(data_test) - i) - 3600 * (times.avg * (len(data_test) - i) // 3600)) // 60)) + 'm')

    text2write = text2write + f'{dataset}: AUC={auc_test} loss={loss}' + '\n'

timestr = str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4])
model_name_extracted = model_param_adr.split("\\")[-1]
file_name = os.path.join('evaluations', f'{model_name_extracted}_eval_{timestr}.txt')
with open(file_name, 'w') as file:
    # Write content to the file
    file.write(text2write)






