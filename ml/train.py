import torch


def train_batch_GAN(sconfig, net_G, net_D, optimizer_G, optimizer_D, real_A, real_B, criterion_gan, criterion_l1):

    # forward pass
    # generate fake image using generator
    fake_B = net_G(real_A)

    ###
    # DISCRMINATOR
    ###
    set_requires_grad(net_D, True)
    optimizer_D.zero_grad()

    # discrminate fake image
    fake_AB = torch.cat((real_A, fake_B), dim=1)  # conditionalGAN takes both real and fake image
    pred_fake = net_D(fake_AB.detach())
    loss_D_fake = criterion_gan(pred_fake, False)

    # discrminate real image
    real_AB = torch.cat((real_A, real_B), dim=1)
    pred_real = net_D(real_AB)
    loss_D_real = criterion_gan(pred_real, True)

    # backward & optimize
    loss_D = (loss_D_fake + loss_D_real) * sconfig.d_loss_factor
    loss_D.backward()
    optimizer_D.step()

    ###
    # GENERATOR
    ###
    set_requires_grad(net_D, False)
    optimizer_G.zero_grad()

    # generator should fool the discriminator
    fake_AB = torch.cat((real_A, fake_B), dim=1)
    pred_fake = net_D(fake_AB)
    loss_G_fake = criterion_gan(pred_fake, True)

    # l1 loss between generated and real image for more accurate output
    loss_G_l1 = criterion_l1(fake_B, real_B) * sconfig.l1_lambda

    # backward & optimize
    loss_G = loss_G_fake + loss_G_l1
    loss_G.backward()
    optimizer_G.step()

    return loss_G_fake.item(), loss_G_l1.item(), loss_D.item()


def set_requires_grad(net, requires_grad):
    for param in net.parameters():
        param.requires_grad = requires_grad