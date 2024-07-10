import torch
import pandas as pd
import random
import time
import os
from torchvision import transforms
import numpy as np
from models import DeepfakeDetector
import torchfunc
from data_loader_pictures import DeepFakeDataset
from double_unet import Double_UNet
from utils import save_result_image, select_random_images_multiple_folders,\
    select_random_images_folder, load_images_from_adr, AverageMeter
from contextlib import contextmanager
import sys
from sklearn.metrics import roc_auc_score
from ae_models import U_Net



# TODO: FIND BEST MODEL IF STABLIZATION NOT POSSIBLE

class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for file in self.files:
            file.write(obj)

    def flush(self):
        for file in self.files:
            file.flush()

@contextmanager
def tee_stdout_and_file(file):
    original_stdout = sys.stdout
    try:
        sys.stdout = Tee(sys.stdout, file)
        yield sys.stdout
    finally:
        sys.stdout = original_stdout


def find_stabilization_point(d_losses, g_losses, threshold=0.05, window_size=5):

    #Find the point where the losses start to stabilize.

    def calculate_avg_change(losses):
        return sum(abs(losses[i] - losses[i - 1]) for i in range(1, len(losses))) / (len(losses) - 1)

    for ep in range(window_size, len(d_losses)):
        d_avg_change = calculate_avg_change(d_losses[ep - window_size:ep])
        g_avg_change = calculate_avg_change(g_losses[ep - window_size:ep])

        if d_avg_change < threshold and g_avg_change < threshold:
            return ep - window_size + 1  # Return the epoch where stabilization starts

    # If stabilization is not found, return -1
    return -1


with open('prints\output' + '_' + str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(
                                  time.gmtime()[3]) + str(time.gmtime()[4]) + '.txt', 'a') as f:
    with tee_stdout_and_file(f):

        torchfunc.cuda.reset()
        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        #######################
        ### PARAMETERS ###
        ######################
        lr = {'discriminator': 1e-5,  #1e-5
              'generator': 1e-3,
              'deepfake_detector': 1e-5}
        n_generator = 5
        n_discriminator = 1
        lambda_deepfake_penalize_initialize = 1
        lambda_mse = 300
        discriminator_noise_level = 0
        #####################
        weight_decay = 0
        nr_epochs = 100
        lr_decay = 0.9
        test_data_frequency = 1
        train_batch_size = 7
        test_batch_size = 8
        gradient_clipping_value = None
        model_param_adr = r'F:\saved_model\deepfake_detector_global_epoch_1000000_param_FF++_143_1941.pkl'# "F:\saved_model\deepfake_detector_global_epoch_46_param_FF++_133_1434.pkl" #r'D:\saved_model\Xception_fullface_epoch_77_param_FF++_33_1446.pkl' # r"F:\saved_model\deepfake_detector_global_epoch_1027_param_FF++_103_1531.pkl"   # None if new training
        discriminator_param_adr = r"F:\saved_model\discriminator_epoch_6_global_epoch_10002_param_FF++_133_2240.pkl" #r'F:\saved_model\discriminator_epoch_9_global_epoch_2_param_FF++_103_2114.pkl'
        generator_param_adr = "F:\saved_model\generator_epoch_6_global_epoch_10002_param_FF++_133_2240.pkl" #r"F:\saved_model\generator_epoch_9_global_epoch_2_param_FF++_103_2114.pkl" #r'F:\saved_model\generator_epoch_6_global_epoch_19_param_FF++_93_1454.pkl' #r'D:\saved_model\new_AE__fullface_epoch_4_param_FF++_132_2220.pkl' #r'F:\saved_model_new\AE\AttuNetBIG_epoch_68_param_FF++_2210_1732.pkl'
        ###########################
        ### DATASET PARAMETERS ###
        ###########################
        dataset_adr = r'D:\ff++\saved_images'
        train_file_path = r'train_test_split.xlsx' #r'train_test_combined_final_modified.xlsx' #r'train_test_split.xlsx'
        img_type = None #'fullface'
        dataset = 'FF++' #celebDF' #'FF++'
        ####################################
        ### TRAIN STOP PARAMETERS ##########
        ####################################
        skip_first_train_generator = True
        mse_target = 0.0015 #0.0015
        deepfake_loss_target = 0.1
        gan_stabilize_threshold = 0.05
        gan_stabilize_mean_window = 4
        epochs_after_conditions_met = 4
        max_epochs = 7
        generated_data_folder = r'F:\saved_generated_data\try5'
        ####################################
        ### DEEPFAKE TRAIN PARAMETERS ##########
        ####################################
        save_df_each_epoch = True
        epoch_done = 999999
        deepfake_train_max_epochs = 30
        real_data_auc_target = 0.98
        #generated_images_auc_target = 0.95
        #new_folder_auc_target = 0.95
        epochs_after_conditions_met_deepfake_train = 30
        deepfake_train_epoch_length = 2000
        deepfake_train_batch_size = 16
        deepfake_weight_decay = 0#0.0001
        # lambda_loss_real = 5
        #############################################


        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #transforms.Normalize([0.5, 0., 0.5], [0.5, 0.5, 0.5])
        ])

        transf_tensor = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        p = 0.25
        transf_augmentations = transforms.Compose([
            transforms.RandomApply([transforms.RandomRotation(15)], p=p),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(1), saturation=(0.5, 1.5), hue=(-0.1, 0.1))],
                p=p),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3), sigma=(10, 25)),
                                    transforms.GaussianBlur(kernel_size=(9, 9), sigma=(10, 25))], p=p),
            transforms.RandomApply([transforms.RandomResizedCrop(size=(299, 299), scale=(0.8, 1))], p=p),
            transforms.RandomAdjustSharpness(2, p=p),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        data_train = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf_tensor,
                                     batch_size=train_batch_size, train='all', image_type=img_type, dataset=dataset, classes_used=[0])
        data_train_reals = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf_tensor,
                                     batch_size=train_batch_size, train=True, image_type=img_type, dataset=dataset, classes_used=[0])
        data_test_reals = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf_tensor,
                                     batch_size=train_batch_size, train=False, image_type=img_type, dataset=dataset, classes_used=[0])
        data_train_deepfake = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf_tensor,
                                              batch_size=deepfake_train_batch_size, train=True, image_type=img_type, dataset=dataset)
        data_train_deepfake_reals = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf_tensor,
                                              batch_size=int(deepfake_train_batch_size//2), train=True, image_type=img_type, dataset=dataset, classes_used=[0])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: ', device)

        # Initialize generator and discriminator
        generator = Double_UNet(output=2)
        #generator = U_Net()
        generator.load_state_dict(torch.load(generator_param_adr))
        discriminator = DeepfakeDetector(pretrained=True, finetuning=False, architecture='Xception')
        if discriminator_param_adr is not None:
            discriminator.load_state_dict(torch.load(discriminator_param_adr))
        deepfake_detector = DeepfakeDetector(pretrained=True, finetuning=False, architecture='Xception')
        deepfake_detector.load_state_dict(torch.load(model_param_adr))

        if cuda:
            generator.cuda()
            discriminator.cuda()
            deepfake_detector.cuda()

        adversarial_loss = torch.nn.BCELoss()
        deepfake_loss = torch.nn.BCELoss()
        mse_loss = torch.nn.MSELoss()

        # ----------- ----------- ----------- ----------- ----------- ----------- ----------- ----------- -----------
        #  Training
        # ----------- ----------- ----------- ----------- ----------- ----------- ----------- ----------- -----------
        deepfake_train_epoch = epoch_done
        while True:

            deepfake_train_epoch += 1

            # Optimizers
            generator_optimizer = torch.optim.Adam(list(generator.parameters()), lr=lr['generator'],
                                                   weight_decay=weight_decay)
            discriminator_optimizer = torch.optim.Adam(list(discriminator.parameters()), lr=lr['discriminator'],
                                                       weight_decay=weight_decay)
            deepfake_detector_optimizer = torch.optim.Adam(list(deepfake_detector.parameters()),
                                                           lr=lr['deepfake_detector'], weight_decay=deepfake_weight_decay)

            if not skip_first_train_generator:
                print('Train generator')
                batches_done = 0
                saved_imgs = []
                g_losses_mean_epoch = []
                d_losses_mean_epoch = []
                for epoch in range(1, max_epochs):
                    print('Epoch ' + str(epoch) + ' training...', end=' ')
                    start = time.time()

                    generator.train()
                    discriminator.train()
                    deepfake_detector.eval()

                    data_train.shuffle()

                    stab = -1

                    losses_d = []
                    losses_g = []
                    losses_g_gan = []
                    losses_deepfake_detector = []
                    mse_losses = []

                    # if epoch > 50:
                    #     lambda_deepfake_penalize = (epoch-50) * lambda_deepfake_penalize_initialize if epoch < 10 else 1
                    # else:
                    #     lambda_deepfake_penalize = 0

                    lambda_deepfake_penalize = lambda_deepfake_penalize_initialize

                    for i in range(int(len(data_train))):

                        t = time.time()
                        real_imgs, targets = data_train[i]
                        if random.random() > 0.25:  # 25% not to augment
                            real_imgs = transf_augmentations(real_imgs)
                        else:
                            real_imgs = normalize(real_imgs)

                        real_imgs, targets = real_imgs.to(device), targets.to(device)

                        valid_label = Tensor(real_imgs.size(0), 1).fill_(1.0)  #ones
                        fake_label = Tensor(real_imgs.size(0), 1).fill_(0.0) #zeroes

                        # ---------------------
                        #  Train Discriminator once every n_generator iterations
                        # ---------------------

                        # data goes through generator
                        fake_images = generator(real_imgs)

                        if i % n_generator == 0:

                            # Train Critic
                            fake_out = discriminator(fake_images)
                            real_out = discriminator(real_imgs)

                            if i % ((int(len(data_train)) - 1)//10) == 0:
                                print('++++++++++++\n Discriminator ++++++++++++')
                                print('Fake out:\n', fake_out)
                                print('Real out:\n', real_out)

                            # adding noise
                            real_imgs_noise = real_imgs + (torch.randn(real_imgs.shape) * discriminator_noise_level).cuda()
                            fake_images_noise = fake_images + (torch.randn(fake_images.shape) * discriminator_noise_level).cuda()

                            real_loss = adversarial_loss(discriminator(real_imgs_noise), valid_label)  # real = 1
                            fake_loss = adversarial_loss(discriminator(fake_images_noise), fake_label) # fake = 0

                            d_loss = (real_loss + fake_loss) / 2

                            discriminator_optimizer.zero_grad()
                            d_loss.backward()
                            discriminator_optimizer.step()

                            losses_d.append(d_loss.item())

                        # -----------------
                        #  Train Generator
                        # -----------------
                        fake_images = generator(real_imgs)

                        if i % n_discriminator == 0:

                            # Loss measures generator's ability to fool the discriminator discriminator(fake_images) should output 0 if it works, but generator should make it output 1
                            g_loss = adversarial_loss(discriminator(fake_images), valid_label)
                            mse_loss_g = mse_loss(fake_images, real_imgs)

                            deepfake_detector_results = adversarial_loss(deepfake_detector(fake_images), fake_label)

                            losses_deepfake_detector.append(deepfake_detector_results.item())

                            if i % 1000 == 0:
                                print('fake', deepfake_detector(fake_images))
                                print('real', deepfake_detector(real_imgs))

                            # g_loss is penalized if the generated image is guessed by the deepfake detector.
                            lambda_tensor = torch.Tensor([lambda_deepfake_penalize]).cuda()
                            generator_deepfake_constraint_loss = deepfake_detector_results * lambda_tensor + mse_loss_g * torch.Tensor([lambda_mse]).cuda() + g_loss

                            generator_optimizer.zero_grad()
                            generator_deepfake_constraint_loss.backward()
                            generator_optimizer.step()

                            losses_g.append(generator_deepfake_constraint_loss.item())
                            losses_g_gan.append(g_loss.item())
                            mse_losses.append(mse_loss_g.item())

                        t2 = time.time() - t

                        if i % ((int(len(data_train)) - 1)//10) == 0:
                            print(f'Epoch {epoch}\n ================== \n g_only_loss={np.mean(losses_g_gan)}\n mse_loss_g={np.mean(mse_losses)}\n'
                                  f' df_loss={np.mean(losses_deepfake_detector)}\n G_loss={np.mean(losses_g)}\n D_loss={np.mean(losses_d)}\n',
                                  ' Est time/Epoch: ' + str(int(t2 * (len(data_train)-i) // 3600)) + 'h' +
                                  str(int((t2 * (len(data_train)-i) - 3600 * (t2 * (len(data_train)-i) // 3600)) // 60)) + 'm' + "\n=================="
                                  )

                            timestr = str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4])
                            save_result_image(fake_images[0], os.path.join('saved', f'{timestr}_fake_epoch_{epoch}_superepoch_{deepfake_train_epoch}.png'))
                            save_result_image(real_imgs[0], os.path.join('saved', f'{timestr}_real_epoch_{epoch}_superepoch_{deepfake_train_epoch}.png'))

                    print('Saving models...')
                    # Saving model
                    model_file_name = '_epoch_' + str(epoch) + f'_global_epoch_{deepfake_train_epoch}' \
                                      + '_param_' + dataset + '_' +\
                                      str(time.gmtime()[2]) +\
                                      str(time.gmtime()[1]) +\
                                      '_' + str(time.gmtime()[3]) +\
                                      str(time.gmtime()[4]) + '.pkl'
                    torch.save(generator.state_dict(), os.path.join(r'F:\saved_model', 'generator' + model_file_name))
                    torch.save(discriminator.state_dict(), os.path.join(r'F:\saved_model', 'discriminator' + model_file_name))

                    # determine if we should continue training the generator or not
                    g_only_loss_mean = np.mean(losses_g_gan)
                    mse_loss_mean = np.mean(mse_losses)
                    df_loss_mean = np.mean(losses_deepfake_detector)
                    d_loss_mean = np.mean(losses_d)

                    g_losses_mean_epoch.append(g_only_loss_mean)
                    d_losses_mean_epoch.append(d_loss_mean)

                    if epoch >= gan_stabilize_mean_window:
                        # calculate if G and D are stabilized
                        print('G LOSSES: ', g_losses_mean_epoch)
                        print('D LOSSES: ', d_losses_mean_epoch)
                        print('MSE loss mean: ', mse_loss_mean)
                        print('DF loss mean: ', df_loss_mean)


                        stab = find_stabilization_point(g_losses_mean_epoch, d_losses_mean_epoch,
                                                        window_size=gan_stabilize_mean_window,
                                                        threshold=gan_stabilize_threshold)
                        if stab != -1:
                            print('GEN STABILISED stab=', stab)
                            if epoch - stab >= epochs_after_conditions_met and mse_loss_mean <= mse_target and df_loss_mean <= deepfake_loss_target:
                                print('Stopping train generator at epoch', epoch)
                                break

                # save generated data
                print('Saving generated data...')
                generator.eval()
                try:
                    os.mkdir(os.path.join(generated_data_folder, f'generated_DTE_{deepfake_train_epoch}'))
                except:
                    pass
                #lastest_generated_folder = f'generated_DTE_{deepfake_train_epoch}'
                for i in range(int(len(data_train_reals))):
                    # saving real images
                    real_imgs, _ = data_train_reals[i]
                    real_imgs_n = normalize(real_imgs)  # normalize because data not normalized
                    real_imgs_n = real_imgs_n.to(device)

                    with torch.no_grad():
                        fake_images = generator(real_imgs_n)

                    for image_nr in range(train_batch_size):
                        save_result_image(fake_images[image_nr], os.path.join(generated_data_folder,
                                                                              f'generated_DTE_{deepfake_train_epoch}',
                                                                              f'{dataset}_train_epoch' + '_' +
                                                                              f'{i}_{image_nr}.png'))
                    # SAVING AUGMENTED IMAGES
                    for jii in range(5):
                        real_imgs_aug = transf_augmentations(real_imgs)

                        real_imgs_aug = real_imgs_aug.to(device)

                        with torch.no_grad():
                            fake_images = generator(real_imgs_aug)

                        for image_nr in range(train_batch_size):
                            save_result_image(fake_images[image_nr], os.path.join(generated_data_folder,
                                                                                  f'generated_DTE_{deepfake_train_epoch}',
                                                                                  f'{dataset}_train_epoch' + '_' +
                                                                                  f'{i}_{image_nr}.png'))

                for i in range(int(len(data_test_reals))):

                    real_imgs, _ = data_test_reals[i]
                    real_imgs_n = normalize(real_imgs)  # normalize because data not normalized
                    real_imgs_n = real_imgs_n.to(device)

                    with torch.no_grad():
                        fake_images = generator(real_imgs_n)

                    for image_nr in range(train_batch_size):
                        save_result_image(fake_images[image_nr], os.path.join(generated_data_folder,
                                                                              f'generated_DTE_{deepfake_train_epoch}',
                                                                              f'{dataset}_test_epoch' + '_' +
                                                                              f'{i}_{image_nr}.png'))

                    # SAVING AUGMENTED IMAGES
                    for jii in range(5):
                        real_imgs_aug = transf_augmentations(real_imgs)

                        real_imgs_aug = real_imgs_aug.to(device)

                        with torch.no_grad():
                            fake_images = generator(real_imgs_aug)

                        for image_nr in range(train_batch_size):
                            save_result_image(fake_images[image_nr], os.path.join(generated_data_folder,
                                                                                  f'generated_DTE_{deepfake_train_epoch}',
                                                                                  f'{dataset}_test_epoch' + '_' +
                                                                                  f'{i}_{image_nr}.png'))

            else:
                skip_first_train_generator = False

            lastest_generated_folder = f'generated_DTE_{deepfake_train_epoch}'

            ####################################################################################################
            ###############################TRAINING DEEPFAKE MODEL##############################################
            ####################################################################################################
            print('Training deepfake model')
            epochs_left = None
            for epoch in range(0, deepfake_train_max_epochs + 1):

                print('Epoch ' + str(epoch) + ' training...')
                start = time.time()

                deepfake_detector.train()
                data_train_deepfake.shuffle()

                losses_total = AverageMeter()
                losses = {'real': AverageMeter(),
                          'folders': AverageMeter(),
                          'latest_folder': AverageMeter()}
                predictions_vect = {'real': [],
                                    'folders': [],
                                    'latest_folder': []}
                targets_vect = {'real': [],
                                'folders': [],
                                'latest_folder': []}

                if epochs_left == 0:
                    print('Finished training')
                    break

                for i in range(deepfake_train_epoch_length):

                    # choose whether to train with real data, generated or new_folder
                    # rng = random.random()
                    # if rng <= 0.5: # train with real data
                    #     data, targets = data_train_deepfake[random.randint(0, int(len(data_train_deepfake)) - 1)]
                    #     data_type_rng = 'real'
                    #
                    # elif rng >= 0.5: # train with data generated in the past epochs
                    #     image_adr = select_random_images_multiple_folders(generated_data_folder,
                    #                                                       num_images=deepfake_train_batch_size//2,
                    #                                                       image_type='train',
                    #                                                       ignore_folder=lastest_generated_folder)
                    #     # getting generated images
                    #     data, targets = load_images_from_adr(image_adr, labels=[1] * (deepfake_train_batch_size//2), transform=transf_tensor)
                    #     # getting real images
                    #     # reals, targets_reals = data_train_deepfake_reals[random.randint(0, int(len(data_train_deepfake_reals)) - 1)]
                    #     # data = torch.cat([data, reals])
                    #     # targets = torch.cat([targets, targets_reals]) #TODO
                    #     data_type_rng = 'folders'

                    # else: # train with the latest generated data
                    #     image_adr = select_random_images_folder(os.path.join(generated_data_folder, lastest_generated_folder),
                    #                                             num_images=deepfake_train_batch_size//2,
                    #                                             image_type='train',)
                    #     # getting generated images
                    #     data, targets = load_images_from_adr(image_adr, labels=[1] * (deepfake_train_batch_size//2), transform=transf_tensor)
                    #     # getting real images
                    #     reals, targets_reals = data_train_deepfake_reals[random.randint(0, int(len(data_train_deepfake_reals)) - 1)]
                    #     data = torch.cat([data, reals])
                    #     targets = torch.cat([targets, targets_reals])
                    #     data_type_rng = 'latest_folder'

                    #rng = random.random()

                    data, targets = data_train_deepfake[random.randint(0, int(len(data_train_deepfake)) - 1)]
                    data = transf_augmentations(data)  # augmenting real data
                    data_type_rng = 'real'


                    image_adr = select_random_images_multiple_folders(generated_data_folder,
                                                                      num_images=deepfake_train_batch_size//2,
                                                                      image_type='train')
                    # getting generated images
                    data_gen, targets_gen = load_images_from_adr(image_adr, labels=[1] * (deepfake_train_batch_size//2), transform=transf)  # generated data is NOT augmented
                    # getting real images
                    reals, targets_reals = data_train_deepfake_reals[random.randint(0, int(len(data_train_deepfake_reals)) - 1)]
                    reals = transf_augmentations(reals)  # augmenting real data
                    data = torch.cat([data, data_gen, reals])
                    targets = torch.cat([targets, targets_gen, targets_reals])

                    data, targets = data.to(device), targets.to(device)

                    outputs_gpu = deepfake_detector(data)

                    outputs = outputs_gpu.to('cpu').flatten()
                    targets = targets.to('cpu')
                    filter_nan = outputs.isnan()
                    outputs = outputs[~filter_nan]
                    targets = targets[~filter_nan]
                    if outputs.shape[0] < train_batch_size:
                        print('problem with nans????????????')
                    if len(targets) == 0:
                        print('BIG problem with nans????????????')
                        continue

                    loss = deepfake_loss(outputs, targets)
                    # if data_type_rng == 'real':
                    #     loss = loss * torch.Tensor([lambda_loss_real]).cuda()
                    deepfake_detector_optimizer.zero_grad()
                    loss.backward()
                    deepfake_detector_optimizer.step()

                    predictions_vect[data_type_rng].append(outputs.detach())
                    targets_vect[data_type_rng].append(targets)
                    losses[data_type_rng].update(loss.item(), data.size(0))
                    losses_total.update(loss.item(), data.size(0))

                    if i % 50 == 49:
                        auc = {'real': None,
                               'folders': None,
                               'latest_folder': None}
                        datatype_vect = ['real']
                        for datatype in datatype_vect:

                            if len(torch.unique(torch.cat(targets_vect[datatype]).flatten())) > 1:
                                auc_train = roc_auc_score(torch.cat(targets_vect[datatype]).flatten(),
                                                          torch.cat(predictions_vect[datatype]).flatten())
                            else:
                                auc_train = '-'

                            auc[datatype] = auc_train

                        print('Minibatch: ' + str(i) + '/' + str(deepfake_train_epoch_length) + f' Epoch {epoch}'
                              '\n Loss total: ' + str(losses_total.avg) +
                              '\n Loss real: ' + str(losses['real'].avg) + ' AUC real: ' + str(auc['real']) +
                              '\n Loss folders: ' + str(losses['folders'].avg) + ' AUC folders: ' + str(auc['folders']) +
                              '\n Loss latest: ' + str(losses['latest_folder'].avg)+ ' AUC latest: ' + str(auc['latest_folder'])
                              + '\n')

                if save_df_each_epoch:
                    print('Finished training deepfake detector')
                    print('Saving models...')
                    # Saving model
                    model_file_name = f'global_epoch_{deepfake_train_epoch}' \
                                      + '_param_' + dataset + '_' + \
                                      str(time.gmtime()[2]) + \
                                      str(time.gmtime()[1]) + \
                                      '_' + str(time.gmtime()[3]) + \
                                      str(time.gmtime()[4]) + '.pkl'
                    torch.save(deepfake_detector.state_dict(),
                               os.path.join(r'F:\saved_model', 'deepfake_detector_' + model_file_name))

                ############################################
                ############# STOPPING TRAIN ###############
                ############################################
                # if auc['folders'] is None:
                #     condition = auc['real'] > real_data_auc_target and auc['latest_folder'] > new_folder_auc_target
                # else:
                #     condition = auc['real'] > real_data_auc_target and auc['folders'] > generated_images_auc_target and auc['latest_folder'] > new_folder_auc_target

                condition = auc['real'] > real_data_auc_target
                if condition:
                    if epochs_left is None:
                        epochs_left = epochs_after_conditions_met_deepfake_train
                    else:
                        epochs_left = epochs_left - 1
                    print(f'\n\n Epoch left: {epochs_left} \n\n')
                else:
                    if epochs_left is not None:
                        print('\n\nEpoch left reset\n\n')
                        epochs_left = None

            print('Finished training deepfake detector')
            print('Saving models...')
            # Saving model
            model_file_name = f'global_epoch_{deepfake_train_epoch}' \
                              + '_param_' + dataset + '_' + \
                              str(time.gmtime()[2]) + \
                              str(time.gmtime()[1]) + \
                              '_' + str(time.gmtime()[3]) + \
                              str(time.gmtime()[4]) + '.pkl'
            torch.save(deepfake_detector.state_dict(), os.path.join(r'F:\saved_model', 'deepfake_detector_' + model_file_name))












