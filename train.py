import argparse
import random

import torch
from torch import nn, optim
from torch.autograd import grad
from torchvision import utils
from tqdm import tqdm

from dataloader import sample_data, lsun_loader, celeba_loader, zi_loader
from model import Discriminator, StyledGenerator


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    """Calculates running average and save average to g_running.
    """
    # In progressive gan paper author used running average of generator when infer samples.
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def train(generator, discriminator, loader, options):
    step = options.init_size // 4 - 1
    data_loader = sample_data(loader, 4 * 2 ** step)
    dataset = iter(data_loader)
    pbar = tqdm(range(800000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    one = torch.FloatTensor([1]).cuda()
    mone = one * -1
    iteration = 0

    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, 0.00002 * iteration)

        if iteration > 100000:
            alpha = 0
            iteration = 0
            step += 1

            if step > 6:
                alpha = 1
                step = 6
            data_loader = sample_data(loader, 4 * 2 ** step)
            dataset = iter(data_loader)

        try:
            real_image, label = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, label = next(dataset)

        iteration += 1

        b_size = real_image.size(0)
        real_image = real_image.cuda()
        label = label.cuda()
        real_predict = discriminator(real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        real_predict.backward(mone)

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(4, b_size, args.code_size, device='cuda').chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, args.code_size, device='cuda').chunk(2, 0)
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(
            fake_image, step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        eps = torch.rand(b_size, 1, 1, 1).cuda()
        x_hat = eps * real_image.data + (1 - eps) * fake_image.data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1)**2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val = grad_penalty.item()
        disc_loss_val = (real_predict - fake_predict).item()

        d_optimizer.step()

        if (i + 1) % args.n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)
            predict = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()
            gen_loss_val = loss.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 100 == 0:
            # Inference sample during training
            '''images = []
            for _ in range(5):
                images.append(g_running(
                    torch.randn(5 * 10, args.code_size).cuda(),
                    step=step, alpha=alpha).data.cpu())'''
            images = g_running(torch.randn(5 * 10, args.code_size).cuda(),
                               step=step, alpha=alpha).data.cpu()

            utils.save_image(images, f'sample/{str(i + 1).zfill(6)}.png', nrow=10, normalize=True, range=(-1, 1))

        if (i + 1) % 10000 == 0:
            torch.save(g_running.state_dict(), f'checkpoint/{str(i + 1).zfill(6)}.model')

        state_msg = (f'{i + 1}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
                     f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.3f}')

        pbar.set_description(state_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Style Based GAN')

    parser.add_argument('--code_size', default=512, type=int, help='latent code size')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--n_critic', default=1, type=int, help='how many times the discriminator is trained per G')

    parser.add_argument('--lr', default=0.001,
                        type=float, help='learning rate')
    parser.add_argument('--init_size', default=8, type=int,
                        help='initial image size')
    parser.add_argument('--mixing', action='store_true',
                        help='use mixing regularization')
    parser.add_argument('-d', '--data', default='celeba', type=str,
                        choices=['celeba', 'lsun', 'zi'],
                        help=('Specify dataset. Currently CelebA and LSUN is supported'))
    parser.add_argument('path', type=str, help='path of specified dataset')

    args = parser.parse_args()

    generator = nn.DataParallel(StyledGenerator(args.code_size)).cuda()
    discriminator = nn.DataParallel(Discriminator()).cuda()
    g_running = StyledGenerator(args.code_size).cuda()
    g_running.train(False)

    class_loss = nn.CrossEntropyLoss()

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group(
        {'params': generator.module.style.parameters(), 'lr': args.lr * 0.01})
    d_optimizer = optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator.module, 0)

    if args.data == 'celeba':
        loader = celeba_loader(args.path, args.batch_size)
    elif args.data == 'lsun':
        loader = lsun_loader(args.path, args.batch_size)
    elif args.data == 'zi':
        loader = zi_loader(args.path, args.batch_size)

    train(generator, discriminator, loader, args)
