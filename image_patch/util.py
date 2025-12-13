import numpy as np
import os
import math
import glob
import random

random.seed(9001)
from random import randint, shuffle
from PIL import Image
import skvideo.io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset
import torchvision.transforms as T
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
import time

from log_utils import setup_logger

logger = setup_logger(__name__)


class MLP(nn.Module):
    """dense net as a function approximator
    h_sizes: list of hidden layer sizes
    out_size: output size
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        layer_sizes=[200, 50],
        activation_func=torch.sigmoid,
    ):
        super(MLP, self).__init__()
        if len(layer_sizes) > 0:
            self.hidden = nn.ModuleList()
            self.hidden.append(nn.Linear(input_dim, layer_sizes[0]))
            for k in range(len(layer_sizes) - 1):
                self.hidden.append(nn.Linear(layer_sizes[k], layer_sizes[k + 1]))
            self.out = nn.Linear(layer_sizes[-1], output_dim)
        else:
            self.hidden = []
            self.out = nn.Linear(input_dim, output_dim)
        self.activation_func = activation_func

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        output = self.activation_func(self.out(x))
        return output


class ParallelLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        out_channels,
        bias=True,
        input_has_channel_dim=True,
    ):
        super(ParallelLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.out_channels = out_channels
        self.input_has_channel_dim = input_has_channel_dim

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, out_features, in_features)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels, out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        input_shape = input.shape
        if self.input_has_channel_dim:
            assert input_shape[0] == self.out_channels
            # Has channel dimension no need to broadcast
            input_reshaped = input.view(input_shape[0], -1, self.in_features)
        else:
            # Create channel dimension for broadcasting
            input_reshaped = input.reshape(-1, self.in_features)
        # output = torch.einsum('coi,cbi->cbo', self.weight, input_reshaped)
        output = torch.matmul(input_reshaped, self.weight.transpose(-1, -2))
        if self.bias is not None:
            # Create batch dimesnion and broadcast
            output += self.bias.unsqueeze(1)
        if self.input_has_channel_dim:
            output_shape = input_shape[:-1] + (self.out_features,)
        else:
            output_shape = (
                (self.out_channels,) + input_shape[:-1] + (self.out_features,)
            )
        return output.view(output_shape)

    def extra_repr(self):
        return "in_features={}, out_features={}, out_channels={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.out_channels,
            self.bias is not None,
        )


class ParallelMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        layer_sizes=[6, 4],
        layer_bias=True,
        input_has_channel_dim=False,
        activation_func=torch.sigmoid,
    ):
        super(ParallelMLP, self).__init__()
        if len(layer_sizes) > 0:
            self.hidden = nn.ModuleList()
            if isinstance(layer_bias, list):
                self.hidden.append(
                    ParallelLinear(
                        input_dim,
                        layer_sizes[0],
                        output_dim,
                        input_has_channel_dim=input_has_channel_dim,
                        bias=layer_bias[0],
                    )
                )  # First layer
            else:
                self.hidden.append(
                    ParallelLinear(
                        input_dim,
                        layer_sizes[0],
                        output_dim,
                        input_has_channel_dim=input_has_channel_dim,
                    )
                )  # First layer
            for k in range(len(layer_sizes) - 1):
                if isinstance(layer_bias, list):
                    self.hidden.append(
                        ParallelLinear(
                            layer_sizes[k],
                            layer_sizes[k + 1],
                            output_dim,
                            bias=layer_bias[k],
                        )
                    )
                else:
                    self.hidden.append(
                        ParallelLinear(layer_sizes[k], layer_sizes[k + 1], output_dim)
                    )
            self.out = ParallelLinear(layer_sizes[-1], 1, output_dim)
        else:
            self.hidden = []
            self.out = ParallelLinear(
                input_dim, 1, output_dim, input_has_channel_dim=input_has_channel_dim
            )
        self.activation_func = activation_func

    def forward(self, x):
        for layer in self.hidden:
            x = F.relu(layer(x))
        output = self.activation_func(self.out(x))
        output = output.squeeze(
            -1
        )  # remove the output feature dimension as now we replaced it with the output channel dimension
        num_dims = output.dim()
        permute_order = list(range(1, num_dims)) + [0]
        output = output.permute(*permute_order)
        return output


class MLPConv2d(nn.Module):
    """Layer to create sparse representation with input and output like conv2d
    funapp: function approximation layer. (e.g. MLP network)
    """

    def __init__(
        self,
        layer_sizes,
        in_channels,
        out_channels,
        kernel_size,
        stride=None,
        layer_bias=None,
    ):
        super(MLPConv2d, self).__init__()
        self.funapp = MLP(
            input_dim=in_channels * np.prod(kernel_size),
            output_dim=out_channels,
            layer_sizes=layer_sizes,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert self.out_channels == self.funapp.out.out_features
        self.kernel_size = kernel_size
        if not stride:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        b, c, h, w = x.shape
        h_out = int((h - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        x = self.unfold(x).transpose(1, 2)
        x = self.funapp(x).transpose(1, 2)
        out = F.fold(x, (h_out, w_out), (1, 1))
        # print(out.shape)
        return out


class IndependentMLPConv2d(nn.Module):
    """Similary to MLPConv2d except each node has dedicated MLP."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=None,
        funapp=None,
        layer_sizes=None,
        layer_bias=None,
    ):
        super(IndependentMLPConv2d, self).__init__()
        if funapp is not None and layer_sizes is not None:
            raise TypeError("Pass only one of 'funapp' or 'layer_sizes'.")

        if funapp is None:
            funapp = layer_sizes

        if isinstance(funapp, list):
            self.funapp_list = nn.ModuleList()
            for i in range(out_channels):
                self.funapp_list.append(
                    MLP(
                        input_dim=in_channels * np.prod(kernel_size),
                        output_dim=1,
                        layer_sizes=funapp,
                    )
                )
        else:
            raise Exception("Not implemented.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        # assert(self.out_channels == self.funapp.out.out_features)
        self.kernel_size = kernel_size
        if not stride:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        b, c, h, w = x.shape
        h_out = int((h - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        x = self.unfold(x).transpose(1, 2)
        outputs = []
        for i in range(self.out_channels):
            outputs.append(self.funapp_list[i](x))
        x = torch.cat(outputs, dim=-1)
        x = x.transpose(1, 2)
        out = F.fold(x, (h_out, w_out), (1, 1))
        # print(out.shape)
        return out


class ParallelMLPConv2d(nn.Module):
    """
    Similar to IndependentMLPConv2d except Parallelized and 10x faster.
    """

    def __init__(
        self,
        layer_sizes,
        layer_bias,
        in_channels,
        out_channels,
        kernel_size,
        stride=None,
    ):
        super(ParallelMLPConv2d, self).__init__()
        if isinstance(layer_sizes, list):
            self.funapp = ParallelMLP(
                input_dim=in_channels * np.prod(kernel_size),
                output_dim=out_channels,
                layer_sizes=layer_sizes,
                layer_bias=layer_bias,
            )
        else:
            raise Exception("Not implemented.")
        self.in_channels = in_channels
        self.out_channels = out_channels
        # assert(self.out_channels == self.funapp.out.out_features)
        self.kernel_size = kernel_size
        if not stride:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        b, c, h, w = x.shape
        h_out = int((h - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        w_out = int((w - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        x = self.unfold(x).transpose(
            1, 2
        )  # x shape after this step is b, h_out*w_out, c*h*w
        x = self.funapp(x)
        x = x.transpose(1, 2)
        out = F.fold(x, (h_out, w_out), (1, 1))
        return out


class Round(nn.Module):
    """Round layer."""

    def __init__(self):
        super(Round, self).__init__()

    def forward(self, x):
        return torch.round(x)


class EvenCodeNet(nn.Module):
    def __init__(self, config):
        super(EvenCodeNet, self).__init__()
        self.layers = nn.ModuleList()
        for layer_cg in config.layers:
            layer_type = layer_cg.pop("type")
            requires_grad = layer_cg.pop("requires_grad")
            pretrained = layer_cg.pop("pretrained")
            if layer_type in [
                "ParallelMLPConv2d",
                "IndependentMLPConv2d",
                "MLPConv2d",
            ]:
                # Search for the layer class in both torch.nn and custom modules
                LayerClass = getattr(nn, layer_type, None) or globals().get(layer_type)
            if LayerClass is None:
                raise ValueError(
                    f"Layer type {layer_type} not found in torch.nn or custom modules"
                )
            layer = LayerClass(**layer_cg)
            if pretrained:
                layer, _, _, _ = load_model(pretrained, layer)
            layer.requires_grad_(requires_grad)
            self.layers.append(layer)

    def forward(self, x, y=None):
        for layer in self.layers:
            x = layer(x)
        return x


class ImagePatchesWithLabel(Dataset):
    def __init__(
        self,
        base_dataset,
        patch_height=4,
        patch_width=4,
        num_patches_per_file=1000,
        min_image_size=64,
        transforms=None,
    ):
        self.base_dataset = base_dataset
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.num_patches_per_file = num_patches_per_file
        self.min_image_size = min_image_size
        self.transforms = transforms
        random.seed(123)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        C, H, W = img.shape

        if W < self.min_image_size or H < self.min_image_size:
            # Skip this sample and pick another one
            # (e.g. randomly choose a new idx or raise an exception)
            return self.__getitem__((idx + 1) % len(self))

        # # Resize if needed
        # scale_factor = 2 / min(H / self.patch_height, W / self.patch_width)
        # if scale_factor > 1:
        #     new_height = int(H * scale_factor)
        #     new_width = int(W * scale_factor)
        #     img = T.functional.resize(img, (new_height, new_width), antialias=True)
        #     _, H, W = img.shape

        # Vectorized random patch extraction
        rows = torch.randint(0, H - self.patch_height + 1, (self.num_patches_per_file,))
        cols = torch.randint(0, W - self.patch_width + 1, (self.num_patches_per_file,))
        patches = torch.stack(
            [
                img[:, r : r + self.patch_height, c : c + self.patch_width]
                for r, c in zip(rows, cols)
            ]
        )
        labels = torch.full((self.num_patches_per_file,), label, dtype=torch.long)

        if self.transforms:
            patches = self.transforms(patches)

        return patches, labels


class ImageFolder(VisionDataset):
    def __init__(
        self, root, transform=None, target_transform=None, loader=default_loader
    ):
        super(ImageFolder, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.loader = loader
        self.extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        )

        # Get image file paths
        self.imgs = self._get_imgs(self.root)

    def _get_imgs(self, dir):
        images = []
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if fname.lower().endswith(self.extensions):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is always 0.
        """
        path = self.imgs[index]
        target = 0  # always return 0 as class label

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class VideoPatches(Dataset):
    def __init__(
        self,
        video_folder,
        patch_height=4,
        patch_width=4,
        patch_frame=4,
        num_patches_per_file=1000,
        as_gray=False,
    ):
        super(VideoPatches).__init__()
        self.video_folder = video_folder
        self.video_paths = [
            fn for fn in glob.iglob(video_folder + "**/*.mp4", recursive=True)
        ]
        logger.info(f"Totoal number of images {len(self.video_paths)}")
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.patch_frame = patch_frame
        self.num_patches_per_file = num_patches_per_file
        self.as_gray = as_gray

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        patches = []
        filename = self.video_paths[idx]
        # print(filename)
        vid = skvideo.io.vread(filename, as_grey=self.as_gray)
        vid = vid.astype(np.float32)
        vid = vid / 255
        vid = vid.transpose((0, 3, 1, 2))
        F, C, H, W = vid.shape
        for patch_index in range(self.num_patches_per_file):
            row = randint(0, H - self.patch_height)
            col = randint(0, W - self.patch_width)
            try:
                frame = randint(0, F - self.patch_frame)
            except:
                print(filename)
                print(F, C, H, W)
                print(self.patch_frame)
                print(row, col)
                return
            patch = vid[
                frame : frame + self.patch_frame,
                :,
                row : row + self.patch_height,
                col : col + self.patch_width,
            ]
            patches.append(patch)
        patches = np.stack(patches)
        return patches


def save_model(model_save_path, model, optimizer, loss_history, scheduler=None):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_history,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        },
        model_save_path,
    )


def load_model(model_save_path, model, optimizer=None, scheduler=None, device=None):
    # Load the model
    print(f"Loading model from {model_save_path}")
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    loss_history = checkpoint["loss_history"]
    return model, optimizer, loss_history, scheduler


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`

    Ref: https://stackoverflow.com/a/62764464/3778898
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)
