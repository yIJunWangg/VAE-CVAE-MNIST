import os
import time
import torch
import argparse
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.nn.functional as F

import dataloaders.dataloaders as dataloaders
from models import VAE
import seaborn as sns
import matplotlib.pyplot as plt

# def to_onehot(label_map, num_classes=5):
#     # label_map: [B, 1, H, W] -> one-hot: [B, num_classes, H, W]
#     return F.one_hot(label_map.squeeze(1).long(), num_classes).permute(0, 3, 1, 2).float()

def loss_fn(recon_x, x, mean, log_var):
    BCE = F.cross_entropy(recon_x.reshape(-1,512*512), x.reshape(-1,512*512), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return (BCE + KLD) / x.size(0)

def main(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()
    dataloader, _ = dataloaders.get_dataloaders(args)

    input_dim = 512 * 512
    vae = VAE(
        encoder_layer_sizes=[input_dim, 2048, 512],
        latent_size=args.latent_size,
        decoder_layer_sizes=[512, 2048, input_dim],
        conditional=args.conditional,
        num_labels=5 if args.conditional else 0).to(device)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)
    logs = defaultdict(list)

    for epoch in range(args.epochs):
        tracker_epoch = defaultdict(lambda: defaultdict(dict))
        for iteration, batch in enumerate(dataloader):
            x = batch['label_xpl']         # [B,1,512,512]
            y = batch['sem_cond'] if args.conditional else None
            # x = to_onehot(x, num_classes=5).view(x.size(0), -1)  # [B, 5*512*512]
            x, y = x.to(device), y.to(device) if y is not None else None

            recon_x, mean, log_var, z = vae(x, y) if args.conditional else vae(x)
            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0 or iteration == len(dataloader)-1:
                print(f"Epoch {epoch:02d}/{args.epochs} Batch {iteration:04d}/{len(dataloader)-1}, Loss {loss.item():9.4f}")
        # if iteration % args.print_every == 0 or iteration == len(dataloader)-1:
        #         print("Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
        #             epoch, args.epochs, iteration, len(dataloader)-1, loss.item()))

        #         if args.conditional:
        #             condition = torch.rand(10,5)
        #             condition = condition / condition.sum()  # Normalize to sum to 1
        #             z = torch.randn([10, args.latent_size]).to(device)
        #             x = vae.inference(z, condition)
        #         else:
        #             z = torch.randn([10, args.latent_size]).to(device)
        #             x = vae.inference(z)

        #         plt.figure()
        #         plt.figure(figsize=(5, 10))
        #         for p in range(10):
        #             plt.subplot(5, 2, p+1)
        #             if args.conditional:
        #                 plt.text(
        #                     0, 0, "c={:d}".format(condition[p].item()), color='black',
        #                     backgroundcolor='white', fontsize=8)
        #             plt.imshow(x[p].view(512, 512).cpu().data.numpy())
        #             plt.axis('off')

        #         if not os.path.exists(os.path.join(args.fig_root, str(ts))):
        #             if not(os.path.exists(os.path.join(args.fig_root))):
        #                 os.mkdir(os.path.join(args.fig_root))
        #             os.mkdir(os.path.join(args.fig_root, str(ts)))

        #         plt.savefig(
        #             os.path.join(args.fig_root, str(ts),
        #                          "E{:d}I{:d}.png".format(epoch, iteration)),
        #             dpi=300)
        #         plt.clf()
        #         plt.close('all')

        # df = pd.DataFrame.from_dict(tracker_epoch, orient='index')
        # g = sns.lmplot(
        #     x='x', y='y', hue='label', data=df.groupby('label').head(100),
        #     fit_reg=False, legend=True)
        # g.savefig(os.path.join(
        #     args.fig_root, str(ts), "E{:d}-Dist.png".format(epoch)),
        #     dpi=300)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--latent_size", type=int, default=512)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--conditional", default=True)
    parser.add_argument("--dataroot", type=str, default='/mnt/windows_F/wyj_project/PetroSynthGAN/datasets/rock')
    parser.add_argument("--name", type=str, default='custom')
    parser.add_argument("--dataset_mode", type=str, default='custom')
    parser.add_argument("--class_dir", type=str, default='/mnt/windows_F/wyj_project/PetroSynthGAN/datasets/rock/class.txt')
    parser.add_argument("--phase", type=str, default='train')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_semantics", type=int, default=5)
    parser.add_argument("--label_unknown", type=int, default=0)
    parser.add_argument('--load_size', type=int, default=512)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--no_flip', action='store_true')
    args = parser.parse_args()

    main(args)
