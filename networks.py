import torch
import torch.nn as nn

#############################
# Generator
#############################

class Generator(nn.Module):
    def __init__(self, ngf=64):
        super(Generator, self).__init__()

        self.ngf = ngf

        self.embedding_input = nn.Sequential(
            nn.ConvTranspose2d(100, self.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4)
        )

        self.embedding_label = nn.Sequential(
            nn.ConvTranspose2d(10, self.ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 4)
        )

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input, label):
        batch_size = input.size(0)

        label = torch.zeros(batch_size, 10).scatter_(1, label.unsqueeze(1), .1)
        label = label.view(batch_size, 10, 1, 1)

        embed_input = self.embedding_input(input)
        embed_label = self.embedding_label(label)

        embed = torch.cat([embed_input, embed_label], dim=1)

        output = self.generator(embed)

        return output


#############################
# Discriminator
#############################

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()
        self.ndf = ndf

        self.embed_input = nn.Sequential(
            nn.Conv2d(1, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf)
        )

        self.embed_label = nn.Sequential(
            nn.Conv2d(10, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf)
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input, label):
        batch_size = input.size(0)

        label = torch.zeros(batch_size, 10).scatter_(1, label.unsqueeze(1), .1)
        label = label.view(batch_size, 10, 1, 1).repeat(1, 1, 64, 64)

        embed_input = self.embed_input(input)
        embed_label = self.embed_label(label)

        embed = torch.cat([embed_input, embed_label], dim=1)

        return self.discriminator(embed)