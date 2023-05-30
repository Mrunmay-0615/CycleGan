import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from disc import Discriminator
from unet import UNET
from dataset import HorseZebraDataset
from config import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "maps/maps/train"
VAL_DIR = "maps/maps/test"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_H = "genh.pth.tar"
CHECKPOINT_GEN_Z = "genz.pth.tar"
CHECKPOINT_CRITIC_H = "critich.pth.tar"
CHECKPOINT_CRITIC_Z = "criticz.pth.tar"


def train_step(disc_y, disc_x, gen_ytox, gen_xtoy, loader, opt_disc, opt_gen, L1, mse, d_scaler, g_scaler):
    loop = tqdm(loader)
    y_reals = 0
    y_fakes = 0
    for batch_idx, (x, y) in enumerate(loop):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # First train the discriminators
        with torch.cuda.amp.autocast():
            fake_y = gen_xtoy(x)
            dy_fake = disc_y(fake_y.detach())
            dy_real = disc_y(y)
            y_reals += dy_real.mean().item()
            y_fakes += dy_fake.mean().item()
            discy_real_loss = mse(dy_real, torch.ones_like(dy_real))
            discy_fake_loss = mse(dy_fake, torch.zeros_like(dy_fake))
            discy_loss = (discy_real_loss + discy_fake_loss) / 2

            fake_x = gen_ytox(y)
            dx_fake = disc_x(fake_x.detach())
            dx_real = disc_x(x)
            discx_real_loss = mse(dx_real, torch.ones_like(dx_real))
            discx_fake_loss = mse(dx_fake, torch.zeros_like(dx_fake))
            discx_loss = (discx_real_loss + discx_fake_loss) / 2

            D_loss = discy_loss + discx_loss
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generators
        with torch.cuda.amp.autocast():
            # 1. Adversarial Loss
            discx_fake = disc_x(fake_x)
            discy_fake = disc_y(fake_y)
            loss_g_xtoy = mse(discy_fake, torch.ones_like(discy_fake))
            loss_g_ytox = mse(discx_fake, torch.ones_like(discx_fake))
            adv_G_loss = loss_g_ytox + loss_g_xtoy

            # 2. Cycle-consistency loss
            cycle_xtoytox = gen_ytox(fake_y)
            cycle_ytoxtoy = gen_xtoy(fake_x)
            cycle_x_loss = L1(cycle_xtoytox, x)
            cycle_y_loss = L1(cycle_ytoxtoy, y)
            cycle_G_loss = cycle_x_loss + cycle_y_loss

            # 3. Identity loss
            identity_x = gen_ytox(x)
            identity_y = gen_xtoy(y)
            identity_x_loss = L1(identity_x, x)
            identity_y_loss = L1(identity_y, y)
            identity_G_loss = identity_x_loss + identity_y_loss

            # add all togethor
            G_loss = adv_G_loss + LAMBDA_CYCLE * cycle_G_loss + LAMBDA_IDENTITY * identity_G_loss

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if batch_idx % 200 == 0:
            save_image(fake_y * 0.5 + 0.5, f"saved_images/horse_{batch_idx}.png")
            save_image(fake_x * 0.5 + 0.5, f"saved_images/zebra_{batch_idx}.png")

        loop.set_postfix(y_real=y_reals / (batch_idx + 1), y_fake=y_fakes / (batch_idx + 1))


def main():
    disc_x = Discriminator(in_channels=3).to(DEVICE)
    disc_y = Discriminator(in_channels=3).to(DEVICE)
    gen_xtoy = UNET(3, 3).to(DEVICE)
    gen_ytox = UNET(3, 3).to(DEVICE)

    opt_disc = optim.Adam(
        params = list(disc_x.parameters()) + list(disc_y.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )
    opt_gen = optim.Adam(
        params=list(gen_xtoy.parameters()) + list(gen_ytox.parameters()),
        lr=LEARNING_RATE,
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_H,
            gen_xtoy,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_Z,
            gen_ytox,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_H,
            disc_y,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Z,
            disc_x,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = HorseZebraDataset(
        root_horse=config.TRAIN_DIR + "/X",
        root_zebra=config.TRAIN_DIR + "/Y",
        transform=config.transforms,
    )
    val_dataset = HorseZebraDataset(
        root_horse=config.VAL_DIR + "/X",
        root_zebra=config.VAL_DIR + "/Y",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_step(
            disc_y,
            disc_x,
            gen_ytox,
            gen_xtoy,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_xtoy, opt_gen, filename=config.CHECKPOINT_GEN_H)
            save_checkpoint(gen_ytox, opt_gen, filename=config.CHECKPOINT_GEN_Z)
            save_checkpoint(disc_y, opt_disc, filename=config.CHECKPOINT_CRITIC_H)
            save_checkpoint(disc_x, opt_disc, filename=config.CHECKPOINT_CRITIC_Z)


if __name__ == '__main__':
    main()