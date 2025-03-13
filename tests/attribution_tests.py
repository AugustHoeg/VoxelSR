import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import numpy as np

# Define a simple U-Net model
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Create a dummy input and target
input_tensor = torch.randn(1, 1, 64, 64, requires_grad=True)
target_tensor = torch.randint(0, 2, (1, 1, 64, 64)).float()

# Initialize the model and set it to evaluation mode
model = UNet()
model.eval()

# Forward pass
output = model(input_tensor)

# Initialize Integrated Gradients
ig = IntegratedGradients(model)

# Compute attributions
attributions = ig.attribute(input_tensor, target=target_tensor, n_steps=50)

# Convert attributions to numpy for visualization
attributions_np = attributions.detach().numpy()

# Print the shape of the attributions
print("Attributions shape:", attributions_np.shape)

