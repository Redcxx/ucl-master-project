from ml.plot_utils import plot_inp_tar_out


def evaluate_GAN(net_G, net_D, test_dataloader, criterion, device, n_display_samples=3):
    net_G.eval()
    net_D.eval()

    eval_losses = []
    for i, (inp, tar) in enumerate(test_dataloader):
        inp, tar = inp.to(device), tar.to(device)

        out = net_G(inp)
        loss = criterion(out, tar)
        eval_losses.append(loss.item())

        if i < n_display_samples:
            plot_inp_tar_out(inp, tar, out)

    return eval_losses
