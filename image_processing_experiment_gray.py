# Import necessary libraries
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load an image from the file system using the PIL library
image = Image.open("my_love.png")

# Convert the loaded image to grayscale. This will reduce the image from having multiple channels (like RGB or RGBA)
# to just a single channel. This simplifies the image and makes it compatible with our single-channel kernel later on.
image = image.convert("L")

# Define a transformation pipeline using torchvision's transforms.
# In this case, the only transformation we're applying is converting the PIL image to a PyTorch tensor.
transform = transforms.Compose([transforms.ToTensor()])

# Apply the transformation to the image. This converts the image to a tensor.
# The unsqueeze(0) function adds an additional batch dimension to the tensor, making it compatible with many PyTorch functions.
# Finally, .cuda() moves the tensor to the GPU for faster processing.
image_tensor = transform(image).unsqueeze(0).cuda()

# Define a Gaussian blur kernel. This is a 3x3 matrix that, when convolved with an image, will produce a blurred effect.
# The values in the kernel determine the nature and strength of the blur.
kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], device='cuda:0').float()

# Normalize the kernel so that its values sum up to 1. This ensures that the brightness of the image remains consistent after blurring.
kernel /= kernel.sum()

# Apply the Gaussian blur to the image tensor using a convolution operation.
# The kernel is unsqueezed twice to add two additional dimensions, making it compatible with the conv2d function.
# The result is a blurred version of the original image.
blurred_image = torch.nn.functional.conv2d(image_tensor, kernel.unsqueeze(0).unsqueeze(0))

# Convert the blurred tensor back to a PIL image
blurred_image_cpu = blurred_image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)
blurred_image_cpu = blurred_image_cpu.squeeze(-1)  # Remove the third dimension
blurred_image_pil = Image.fromarray((blurred_image_cpu * 255).astype('uint8'))

# Display the blurred grayscale image
blurred_image_pil.show()
