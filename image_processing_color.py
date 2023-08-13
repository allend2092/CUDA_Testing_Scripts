# Import necessary libraries
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load an image from the file system using the PIL library
image = Image.open("my_love.png")

# Define a transformation pipeline using torchvision's transforms.
# In this case, the only transformation we're applying is converting the PIL image to a PyTorch tensor.
transform = transforms.Compose([transforms.ToTensor()])

# Apply the transformation to the image. This converts the image to a tensor.
# The unsqueeze(0) function adds an additional batch dimension to the tensor, making it compatible with many PyTorch functions.
# Finally, .cuda() moves the tensor to the GPU for faster processing.
image_tensor = transform(image).unsqueeze(0).cuda()

# 7x7 Gaussian blur kernel
kernel = torch.tensor([
    [1, 6, 15, 20, 15, 6, 1],
    [6, 36, 90, 120, 90, 36, 6],
    [15, 90, 225, 300, 225, 90, 15],
    [20, 120, 300, 400, 300, 120, 20],
    [15, 90, 225, 300, 225, 90, 15],
    [6, 36, 90, 120, 90, 36, 6],
    [1, 6, 15, 20, 15, 6, 1]
], device='cuda:0').float()

# Normalize the kernel so that its values sum up to 1. This ensures that the brightness of the image remains consistent after blurring.
kernel /= kernel.sum()

# Initialize a tensor to store the results of the blurred channels
blurred_channels = []

# Iterate over each channel in the image tensor and apply the Gaussian blur
for channel in range(image_tensor.shape[1]):
    channel_data = image_tensor[:, channel:channel+1, :, :]
    # Apply the blur multiple times to the current channel
    num_iterations = 5
    for _ in range(num_iterations):
        channel_data = torch.nn.functional.conv2d(channel_data, kernel.unsqueeze(0).unsqueeze(0))

    blurred_channels.append(channel_data)

# Stack the blurred channels back together to get the final blurred image
blurred_image = torch.cat(blurred_channels, dim=1)

# Convert the blurred tensor back to a PIL image
blurred_image_cpu = blurred_image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
blurred_image_pil = Image.fromarray((blurred_image_cpu * 255).astype('uint8'))

# Display the blurred image
blurred_image_pil.show()
