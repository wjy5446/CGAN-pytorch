import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from networks import Generator, Discriminator
from losses import GANLoss
from utils.misc import load_model, save_model

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    num_epoch = 2

    # load dataset and data loader
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST('.', transform=transform, download=True)
    dataloader = data.DataLoader(dataset, batch_size=4)

    # model
    g = Generator()
    d = Discriminator()

    # losses
    gan_loss = GANLoss()

    # use
    is_cuda = torch.cuda.is_available()
    if is_cuda:
        g = g.cuda()
        d = d.cuda()

    # optimizer
    optim_G = optim.Adam(g.parameters())
    optim_D = optim.Adam(d.parameters())

    # train
    for epoch in range(num_epoch):
        total_batch = len(dataloader)

        for idx, (image, label) in enumerate(dataloader):
            d.train()
            g.train()

            # fake image 생성

            noise = torch.randn(4, 100, 1, 1)
            output_fake = g(noise, label)

            # Loss

            d_loss_fake = gan_loss(d(output_fake.detach(), label), False)
            d_loss_real = gan_loss(d(image, label), True)
            d_loss = (d_loss_fake + d_loss_real) / 2

            g_loss = gan_loss(d(output_fake, label), True)

            # update
            optim_G.zero_grad()
            g_loss.backward()
            optim_G.step()

            optim_D.zero_grad()
            d_loss.backward()
            optim_D.step()

            if ((epoch * total_batch) + idx) % 1000 == 0:
                print('Epoch [%d/%d], Iter [%d/%d], D_loss: %.4f, G_loss: %.4f'
                      % (epoch, num_epoch, idx + 1, total_batch, d_loss.item(), g_loss.item()))

                save_model('model', 'GAN', g, {'loss': g_loss.item()})


