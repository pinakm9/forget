
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights

class InceptionV3Features(nn.Module):
    def __init__(self):
        """
        Initialize the InceptionV3 model and remove the final classifier.
        """
        super().__init__()
        weights = Inception_V3_Weights.IMAGENET1K_V1
        inception = inception_v3(weights=weights, aux_logits=True)
        inception.fc = nn.Identity()  # remove classifier
        self.model = inception.eval()

    def forward(self, x):
        # Resize handled externally or internally if needed
        """
        Forward pass to compute InceptionV3 features.

        Parameters:
        x (torch.Tensor): Input tensor of shape (N, C, H, W). 
                          If height (H) or width (W) is not 299, 
                          the input will be resized to (299, 299).

        Returns:
        torch.Tensor: Features extracted by InceptionV3, typically of shape (N, 2048).
        """

        with torch.no_grad():
            if x.shape[-1] != 299 or x.shape[-2] != 299:
                x = nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            return self.model(x)





def compute_inception_features(tensor_imgs, batch_size=64):
    """
    Compute 2048-dim InceptionV3 features from a batch of image tensors.
    tensor_imgs: (N, 3, H, W), values in [-1, 1] or [0, 1]
    """
    device = tensor_imgs.device
    model = InceptionV3Features().to(device)
    
    # Ensure proper normalization
    if tensor_imgs.min() >= 0:
        tensor_imgs = tensor_imgs * 2 - 1  # scale [0,1] → [-1,1]

    features = []
    for i in range(0, tensor_imgs.size(0), batch_size):
        try:
            torch.cuda.empty_cache()
        except:
            pass
        batch = tensor_imgs[i:i + batch_size]
        feats = model(batch)
        features.append(feats)
    return torch.cat(features, dim=0) # Shape: (N, 2048)


    # def compute_features(self, images, identifier, layer=None):
    #     """
    #     Extract features from a single image or batch using a forward hook.

    #     Args:
    #         identifier (nn.Module): identifier instance (e.g., CNN)
    #         images (Tensor): input image tensor, shape [C, H, W] or [B, C, H, W]
    #         layer (nn.Module, optional): target layer to hook. Defaults to identifier.classifier[1].

    #     Returns:
    #         np.ndarray: feature vector(s), shape [1, D] or [B, D]
    #     """
    #     features = []

    #     if layer is None:
    #         layer = identifier.classifier[1]  # default: dense layer before output

    #     def hook_fn(module, input, output):
    #         features.append(output.detach())

    #     hook = layer.register_forward_hook(hook_fn)

    #     with torch.no_grad():
    #         _ = identifier(images)

    #     hook.remove()
    #     return features[0]
    


# def save_tensor_images(tensor, folder, prefix="img"):
#     """
#     Save a batch of images to a given folder.
#     Assumes tensor shape (B, C, H, W) with values in [-1, 1] or [0, 1].
#     """
#     os.makedirs(folder, exist_ok=True)
#     for i, img in enumerate(tensor):
#         save_image(img, os.path.join(folder, f"{prefix}_{i:05d}.png"), normalize=True, value_range=(-1, 1))

# def compute_fid_from_tensors(tensor_real, tensor_fake):
#     """
#     Compute FID between two image tensors using clean-fid, with temp folder cleanup.
#     """
#     with tempfile.TemporaryDirectory() as temp_dir:
#         real_folder = os.path.join(temp_dir, 'real')
#         fake_folder = os.path.join(temp_dir, 'generated')

#         save_tensor_images(tensor_real, real_folder)
#         save_tensor_images(tensor_fake, fake_folder)

#         fid_score = compute_fid(
#             fdir1=real_folder,
#             fdir2=fake_folder,
#             batch_size=256,
#             device='cuda' if torch.cuda.is_available() else 'cpu', num_workers=0
#         )

#     # All folders and files are automatically deleted here
#     return fid_score


# @ut.timer
    # def compute_fid_from_folder(self, real_images, identifier, folder):
    #     device = real_images.device
    #     # find the last checkpoint
    #     checkpoints = glob.glob(''.join([folder, '/checkpoints/*.pth']))
    #     def extract_int(path):
    #         return int(os.path.basename(path).split('_')[-1].split('.')[0])           
            
    #     checkpoints.sort(key=extract_int)
    #     checkpoint = checkpoints[-1]

    #     # load models
    #     model = torch.load(checkpoint, weights_only=False, map_location=device)
    #     model.to(device)

    #     gen_images = model.decode(torch.randn(real_images.shape[0], self.train_kwargs['latent_dim']).to(device))
    #     # compute fid
    #     return self.compute_fid(real_images, gen_images, identifier)
