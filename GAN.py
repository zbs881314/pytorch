import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


BATCH_SIZE = 64
LR_G = 0.0001    # learning rate for generator
LR_D = 0.0001    # learning rate for discriminator
N_IDEAS = 5      # think of this as number of ideas for generating an art work
ART_COMPONENTS = 15   # it could be total point G can draw in the curves
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])

# show our painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()


def artist_works():   # real target
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a - 1)
    paintings = torch.from_numpy(paintings).float()
    return paintings

G = nn.Sequential(             # generator
    nn.Linear(N_IDEAS, 128),    # random ideas
    nn.ReLU(),
    nn.Linear(128, ART_COMPONENTS),     # making a painting from these random ideas
)

D = nn.Sequential(                     # discriminator
    nn.Linear(ART_COMPONENTS, 128),   # receive art work either from the famous artist or a newbie like G
    nn.ReLU(),
    nn.Linear(128, 1),
    nn.Sigmoid(),         # tell the probability that the art work is made by artist
)


opt_D = torch.optim.Adam(D.parameters(), lr=LR_D)
opt_G = torch.optim.Adam(G.parameters(), lr=LR_G)

plt.ion()

for step in range(10000):
    artist_paintings = artist_works()   # real painting from artist
    G_ideas = torch.randn(BATCH_SIZE, N_IDEAS)   # random ideas
    G_paintings = G(G_ideas)                     # fake painting from G

    prob_artist0 = D(artist_paintings)     # D try to increase this prob
    prob_artist1 = D(G_paintings)          # D try to reduce this prob

    D_loss = - torch.mean(torch.log(prob_artist0) + torch.log(1.- prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))

    opt_D.zero_grad()
    D_loss.backward(retain_graph=True)
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

