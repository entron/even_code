# %%
import time
import torch

torch.set_float32_matmul_precision("high")
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR

import util
from config import Config
from log_utils import setup_logger

logger = setup_logger("train")
cg = Config()

input_dim = cg.input_channel * cg.input_height * cg.input_width


net = util.SparseEvenPartitionNet(config=cg)
if cg.compile:
    net = torch.compile(net)
net.to(cg.device)
print(net)

# %%
logger.info(
    f"num_params: {util.numel(net, False)}  num_params_optim: {util.numel(net, True)}"
)


# %%

if cg.as_gray:
    dataset_transform = T.Compose(
        [
            T.Grayscale(),  # Convert RGB images to grayscale
            T.RandomHorizontalFlip(),
            T.ToTensor(),  # Convert PIL image to PyTorch tensor
        ]
    )
else:
    dataset_transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.ToTensor(),  # Convert PIL image to PyTorch tensor
        ]
    )


# base_dataset = datasets.ImageNet(root=cg.image_path, transform=dataset_transform)
base_dataset = util.ImageFolder(root=cg.image_path, transform=dataset_transform)

# In this script we ignore label
train_dataset = util.ImagePatchesWithLabel(
    base_dataset,
    patch_height=cg.input_height,
    patch_width=cg.input_width,
    num_patches_per_file=cg.patches_per_file,
)


# %%

train_dataloder = DataLoader(
    train_dataset,
    batch_size=cg.batch_size,
    shuffle=True,
    num_workers=cg.num_workers,
    pin_memory=True,
    drop_last=True,
)


# %%
n_epoch = cg.n_epoch
mini_batch_size = cg.mini_batch_size

lr = float(cg.lr)
min_lr = float(cg.min_lr)

optimizerClass = getattr(optim, cg.optimizer)
optimizer = optimizerClass(net.parameters(), lr=lr)

loss_history = []

start_epoch = 0
if cg.init_from_epoch:
    model_save_path = cg.fname_checkpoint.format(epoch=cg.init_from_epoch)
    net.layers[-1], optimizer, loss_history, _ = util.load_model(
        model_save_path, net.layers[-1], optimizer
    )
    start_epoch = cg.init_from_epoch + 1

if cg.num_image == -1:
    cg.num_image = len(base_dataset)

num_batches = cg.num_image // cg.batch_size
total_steps = (
    # cg.num_workers  # For iterative dataset each worker will iterate the dataset seperately
    num_batches
    * (n_epoch - start_epoch)
    * (cg.batch_size * cg.patches_per_file // mini_batch_size)
)

scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=min_lr)

net.train()
iteration = 0
running_loss = 0.0
start_time = time.time()
for epoch in range(start_epoch, n_epoch):
    for patches, labels in train_dataloder:
        patches = patches.reshape(-1, *patches.shape[2:])

        if cg.in_batch_shuffle:
            shuffled_index = torch.randperm(len(patches))
            patches = patches[shuffled_index]

        num_mini_batches = len(patches) // mini_batch_size
        if num_mini_batches == 0:
            continue
        for i in range(num_mini_batches):
            data = patches[i * mini_batch_size : (i + 1) * mini_batch_size]
            x = data.to(cg.device)
            optimizer.zero_grad(set_to_none=True)

            outputs = net(x)

            outputs = torch.squeeze(outputs)

            if cg.nodewise:
                distances = F.pdist(outputs.transpose(0, 1), p=1)
                dist_scale_factor = mini_batch_size
            else:
                distances = F.pdist(outputs, p=1)
                dist_scale_factor = cg.layers[-1]["out_channels"]
                distances = distances + 1e-38

            distance_measure = -torch.log(distances / dist_scale_factor).mean()
            mean_activation = outputs.mean()

            loss = cg.sparsity_weight * mean_activation + distance_measure

            loss.backward()
            optimizer.step()
            scheduler.step()

            # print training metrics
            running_loss += loss.item()
            iteration += 1
            if iteration % cg.iteration_checkpoint_interval == 0:
                current_lr = scheduler.get_last_lr()[0]
                log_dict = {
                    "epoch": epoch + 1,
                    "n_epoch": n_epoch,
                    "iteration": iteration,
                    "total_steps": total_steps,
                    "lr": current_lr,
                    "train_loss": running_loss / cg.iteration_checkpoint_interval,
                    "activ": mean_activation.item(),
                    "dist": distance_measure.item(),
                    "time_used": time.time() - start_time,
                }
                logger.info(
                    "Epoch [{epoch}/{n_epoch}], Step [{iteration}/{total_steps}], Loss: {train_loss:.4f}, activ: {activ:.6f}, dist: {dist:.6f}, Time: {time_used:.2f}".format(
                        **log_dict
                    )
                )

                start_time = time.time()
                loss_history.append(running_loss / 100)
                running_loss = 0.0
                util.save_model(
                    cg.fname_checkpoint.format(epoch=epoch),
                    net.layers[-1],
                    optimizer,
                    loss_history,
                )


# %%
mean_distance = distances.mean()
logger.info(f"mean_activation: {mean_activation} mean_distance:{mean_distance}")
