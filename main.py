import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler

rollno = 102303246
param_a = 0.5 * (rollno % 7)
param_b = 0.3 * ((rollno % 5) + 1)

dataset = pd.read_csv("data.csv", encoding="latin1", low_memory=False)

no2_values = dataset["no2"].dropna().to_numpy().reshape(-1, 1)
transformed_output = (no2_values + param_a) * np.sin(param_b * no2_values)

normalizer = RobustScaler()
scaled_output = normalizer.fit_transform(transformed_output)

real_tensor = torch.tensor(scaled_output, dtype=torch.float32)


class GeneratorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, z):
        return self.network(z)


class DiscriminatorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


# setting up models
G = GeneratorNet()
D = DiscriminatorNet()

criterion = nn.BCELoss()  # bce -> binary cross entropy
optimizer_G = optim.Adam(G.parameters(), lr=3e-4)
optimizer_D = optim.Adam(D.parameters(), lr=3e-4)

epochs = 3500
mini_batch = 128
total_samples = real_tensor.shape[0]

# training loop
for step in range(epochs):
    batch_indices = torch.randint(0, total_samples, (mini_batch,))
    real_batch = real_tensor[batch_indices]

    noise_vector = torch.randn(mini_batch, 1)
    fake_batch = G(noise_vector)

    real_labels = torch.ones(mini_batch, 1)
    fake_labels = torch.zeros(mini_batch, 1)

    real_pred = D(real_batch)
    fake_pred = D(fake_batch.detach())

    loss_D = criterion(real_pred, real_labels) + criterion(fake_pred, fake_labels)

    optimizer_D.zero_grad()
    loss_D.backward()
    optimizer_D.step()

    noise_vector = torch.randn(mini_batch, 1)
    generated = G(noise_vector)
    prediction = D(generated)

    loss_G = criterion(prediction, real_labels)

    optimizer_G.zero_grad()
    loss_G.backward()
    optimizer_G.step()

    if step % 500 == 0:
        print(
            f"Step {step} | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}"
        )

with torch.no_grad():
    latent_input = torch.randn(10000, 1)
    fake_samples = G(latent_input).numpy()

restored_fake = normalizer.inverse_transform(fake_samples)

# plots
fig1 = plt.figure()
plt.hist(restored_fake, bins=100, density=True, alpha=0.75)
plt.savefig("histogram.png")
plt.close(fig1)

fig2 = plt.figure()
plt.hist(transformed_output, bins=100, density=True, alpha=0.4, label="Original")
plt.hist(restored_fake, bins=100, density=True, alpha=0.8, label="Generated")
plt.legend()
plt.savefig("gan_comparison.png")
plt.close(fig2)
