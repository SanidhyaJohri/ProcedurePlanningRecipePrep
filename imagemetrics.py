import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from PIL import Image
import torch

def resize_image(image, target_size):
    return image.resize(target_size)

def calculate_scores(image1_path, image2_path):
    # Load images
    img1 = Image.open(image1_path).convert('RGB')
    img2 = Image.open(image2_path).convert('RGB')

    # Resize images to a fixed size
    target_size = (512, 512)  # Set your desired target size
    img1 = resize_image(img1, target_size)
    img2 = resize_image(img2, target_size)

    # Convert images to NumPy arrays
    img1_array = np.array(img1)
    img2_array = np.array(img2)

    # Convert NumPy arrays to PyTorch tensors and add a batch dimension
    # Also, normalize the pixel values to the range [0, 1] (LPIPS expects this range)
    img1_tensor = torch.tensor(img1_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    img2_tensor = torch.tensor(img2_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Move tensors to GPU if available
    if torch.cuda.is_available():
        img1_tensor = img1_tensor.cuda()
        img2_tensor = img2_tensor.cuda()

    # Calculate SSIM score
    # Note: Use skimage's SSIM or another library as needed. This is just for illustration.
    ssim_score = ssim(img1_array, img2_array,win_size=11, channel_axis=-1)

    # Calculate PSNR score
    psnr_score = psnr(img1_array, img2_array)

    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').cuda() if torch.cuda.is_available() else lpips.LPIPS(net='alex')

    # Calculate LPIPS score
    lpips_score = lpips_model(img1_tensor, img2_tensor).item()

    return ssim_score, psnr_score, lpips_score

# Example usage:

image1_path = 'images/orig/1.png'  # Replace with the path to your first image
image2_path = 'images/generated/1.png'  # Replace with the path to your second image

ssim_score, psnr_score, lpips_score = calculate_scores(image1_path, image2_path)
print("SSIM score:", ssim_score)
print("PSNR score:", psnr_score)
print("LPIPS score:", lpips_score)
