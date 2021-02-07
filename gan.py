import argparse
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from datareader import load_images
from neuralnet import Discriminator, Generator
from timeit import default_timer as timer
from torch.utils.tensorboard import SummaryWriter


EXPERIMENT_ID = int(time.time()) # used to create new directories to save results of individual experiments
# directories to save resulst of experiments
DEFAULT_IMG_DIR = 'images/{}'.format(EXPERIMENT_ID)
DEFAULT_TENSORBOARD_DIR = 'tensorboard/{}'.format(EXPERIMENT_ID)

# this will vary in the ProGAN
IMG_SIZE = 128

real_label = 1.0 # for images in training set
fake_label = 0.0 # for generated images

# allow program to be run with different arguments; no arguments -> use defaults
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='/home/datasets/celeba-aligned')
parser.add_argument('--discriminator_model_path')
parser.add_argument('--generator_model_path')
parser.add_argument('--learning_rate', default=0.0002, type=float)
parser.add_argument('--mini_batch_size', default=256, type=int)
parser.add_argument('--num_epochs', default=200, type=int)
parser.add_argument('--save_image_dir', default=DEFAULT_IMG_DIR)
parser.add_argument('--save_model_dir', default='models')
parser.add_argument('--tensorboard_dir', default=DEFAULT_TENSORBOARD_DIR)
args = parser.parse_args()

# create directories for images and tensorboard results
os.mkdir('/home/ioanalazar459/celebGAN/{}'.format(args.save_image_dir))
os.mkdir('/home/ioanalazar459/celebGAN/{}'.format(args.tensorboard_dir))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# set up GAN
d_model = Discriminator().to(device) # discriminator
g_model = Generator().to(device)     # generator

# load pre-trained discriminator if a path to a model is given
if args.discriminator_model_path:
    d_model.load_state_dict(torch.load(args.discriminator_model_path))

# load pre-trained generator if a path to a model is given
if args.generator_model_path:
    g_model.load_state_dict(torch.load(args.generator_model_path))

# optimizer for discriminator and generator is Adam optimizer
d_optim = optim.Adam(d_model.parameters(), lr=args.learning_rate)
g_optim = optim.Adam(g_model.parameters(), lr=args.learning_rate)

# binary cross-entropy loss
criterion = nn.BCELoss()

# batch of latent vectors
fixed_noise = torch.randn(64, 512, 1, 1, device=device)

images = load_images(args.data_dir)

# set up TensorBoard
writer = SummaryWriter(args.tensorboard_dir)
writer.add_graph(d_model, torch.tensor(images[:1], device=device))
writer.add_graph(g_model, fixed_noise)

print('Start training...')
start_time_training = timer()
for epoch in range(args.num_epochs):
    start_time_epoch = timer()

    average_discriminator_real_performance = 0.0 # D(x) -- accuracy of D when given real examples x
    average_discriminator_generated_performance = 0.0 # D(G(x)) -- accuracy of D given generated (fake) examples G(x)
    average_discriminator_loss = 0.0
    average_generator_loss = 0.0

    for i in range(100):
        # 1) Discriminator training step
        d_model.zero_grad()

        # 1.1) Train with mini-batch of real examples
        random_indexes = np.random.choice(len(images), args.mini_batch_size)
        mini_batch = torch.tensor(images[random_indexes], device=device)
        mini_batch_labels = torch.full((args.mini_batch_size,), real_label, dtype=torch.float, device=device)

        # Forward pass
        outputs = d_model(mini_batch)
        loss = criterion(outputs, mini_batch_labels)

        # Backward pass - gradients and loss of D for real examples
        loss.backward()

        # Record some stats
        average_discriminator_loss += loss.item()
        average_discriminator_real_performance += outputs.mean().item() # TODO: understand this

        # 1.2) Train with mini-batch of generated (fake) examples
        noise = torch.randn(args.mini_batch_size, 512, 1, 1, device = device)
        mini_batch = g_model(noise) # output: mini_batch_size x (3 x 128 x 128)
        mini_batch_labels.fill_(fake_label) # fills tensor with specified value

        # Forward pass
        # size of outputs: mini_batch_size x 1
        outputs = d_model(mini_batch.detach()) # TODO: Why detach?
        loss = criterion(outputs, mini_batch_labels)

        # Backward pass - gradients and loss of D for fake examples
        loss.backward()

        d_optim.step()

        # Record some stats
        average_discriminator_loss += loss.item()
        average_discriminator_generated_performance += outputs.mean().item()

        # 2) Generator training step
        g_model.zero_grad()
        mini_batch_labels.fill_(real_label)

        # Forward pass
        outputs = d_model(mini_batch)
        loss = criterion(outputs, mini_batch_labels)

        # Backward pass
        loss.backward()
        g_optim.step()

        # Record some stats
        average_generator_loss += loss.item()

    time_elapsed_epoch = timer() - start_time_epoch

    average_discriminator_loss /= 100
    average_generator_loss /= 100
    average_discriminator_real_performance /= 100
    average_discriminator_generated_performance /= 100

    print(('Epoch {} - Discriminator Loss: {:.6f} - Generator Loss: {:.6f} - Average D(x): {:.6f} - Average D(G(x)): {:.6f} - Time: {:.3f}s')
        .format(epoch, average_discriminator_loss, average_generator_loss, average_discriminator_real_performance, average_discriminator_generated_performance, time_elapsed_epoch))

    writer.add_scalar('training/generator/loss', average_generator_loss, epoch)
    writer.add_scalar('training/discriminator/loss', average_discriminator_loss, epoch)
    writer.add_scalar('training/discriminator/real_performance', average_discriminator_real_performance, epoch)
    writer.add_scalar('training/discriminator/generated_performance', average_discriminator_generated_performance, epoch)
    writer.add_scalar('training/epoch_duration', time_elapsed_epoch, epoch)

    with torch.no_grad():
        generated_images = g_model(fixed_noise).detach()
    torchvision.utils.save_image(generated_images, '{}/{}-{}x{}.jpg'.format(args.save_image_dir, epoch, IMG_SIZE, IMG_SIZE), padding=2, normalize=True)

time_elapsed_training = timer() - start_time_training
print('Finished training!')

save_d_model_path = '{}/discriminator_{}.pth'.format(args.save_model_dir, time.time())
print('Saving discriminator model as "{}"...'.format(save_d_model_path))
torch.save(d_model.state_dict(), save_d_model_path)

save_g_model_path = '{}/generator_{}.pth'.format(args.save_model_dir, time.time())
print('Saving generator model as "{}"...'.format(save_g_model_path))
torch.save(g_model.state_dict(), save_g_model_path)

print('Saved models.')
