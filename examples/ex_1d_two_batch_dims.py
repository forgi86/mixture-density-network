from argparse import ArgumentParser
import logging

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from src.blocks import MixtureDensityNetwork, NoiseType


def gen_data(n):
    y = np.linspace(-1, 1, n)
    x = 7 * np.sin(5 * y) + 0.5 * y + 0.5 * np.random.randn(*y.shape)
    return x[:,np.newaxis], y[:,np.newaxis]

def plot_data(x, y):
    plt.hist2d(x, y, bins=35)
    plt.xlim(-8, 8)
    plt.ylim(-1, 1)
    plt.axis('off')


if __name__ == "__main__":

    n_components = 3
    num_seq = 32
    seq_len = 16
    nx = 1
    ny = 1


    batch_size = num_seq * seq_len

    plt.ion()
    argparser = ArgumentParser()
    argparser.add_argument("--n-iterations", type=int, default=2000)
    args = argparser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    x, y = gen_data(n=batch_size)
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    x = x.reshape(num_seq, seq_len, nx)
    y = y.reshape(num_seq, seq_len, ny)

    model = MixtureDensityNetwork(nx, ny, n_components=n_components, hidden_dim=50, noise_type=NoiseType.DIAGONAL)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_iterations)

    for i in range(args.n_iterations):
        optimizer.zero_grad()
        loss = model.loss(x, y).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 100 == 0:
            logger.info(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")

    with torch.no_grad():
        y_samples = model.sample(x)

    x = x.reshape(batch_size, nx)
    y = y.reshape(batch_size, ny)
    y_samples = y_samples.reshape(batch_size, ny)

    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plot_data(x[:, 0].numpy(), y[:, 0].numpy())
    plt.title("Observed data")
    plt.subplot(1, 2, 2)
    plot_data(x[:, 0].numpy(), y_samples[:, 0].numpy())
    plt.title("Sampled data")
    plt.show()
