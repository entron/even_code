# %%
import numpy as np
import random
random.seed(9001)
from PIL import Image
import torch
torch.manual_seed(0)
torch.autograd.set_detect_anomaly(True)

from torch.utils.data import DataLoader
import torchvision.transforms as T
import util
import plot_utils
from config import Config
import matplotlib.pyplot as plt


cg = Config()
cg.layers[0]['stride'] = [2, 2]  # reduce GPU RAM usage when calling plot_pca_feature_map()
net = util.SparseEvenPartitionNet(config=cg)
# if cg.compile:
#     net = torch.compile(net)
net.to(cg.device)
print(net)

layer = net.layers[-1]
model_path = cg.fname_checkpoint.format(epoch=cg.n_epoch-1)
layer, _, _, _ = util.load_model(model_path, layer, device=cg.device)
layer.requires_grad_(False)

net.eval()

# Plot training loss history using the saved checkpoint
plot_utils.plot_loss_curve(model_path)

# %%
print(util.numel(net, False), util.numel(net, True))

if cg.as_gray:
    dataset_transform = T.Compose([
        T.Grayscale(),  # Convert RGB images to grayscale
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # Convert PIL image to PyTorch tensor
    ])
else:
    dataset_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor(),  # Convert PIL image to PyTorch tensor
    ])


base_dataset = util.ImageFolder(root=cg.image_path, transform=dataset_transform)

test_dataset = util.ImagePatchesWithLabel(
    base_dataset,
    patch_height=5,
    patch_width=5,
    num_patches_per_file=cg.patches_per_file,
)


test_dataloder = DataLoader(
    test_dataset,
    batch_size=cg.batch_size,
#     shuffle=True,
    num_workers=0,
    pin_memory=True,
    drop_last=True,
)


# %%
img_patches = []

for patches, _ in test_dataloder:
    patches = patches.reshape(-1, *patches.shape[2:])
    img_patches.append(patches)
    if len(img_patches) == 10:
        break

# %%
img_patches = np.concatenate(img_patches)
print(img_patches.shape)

# %%
np.save('test_patches.npy', img_patches)
# img_patches = np.load('/ssd/IMAGES/IMAGENET/test_patches_5x5.npy')


# %%
r = []
mini_batch_size = 1000
num_mini_batches = len(img_patches) // mini_batch_size
print(num_mini_batches)
for i in range(num_mini_batches):
    data = img_patches[i*mini_batch_size:(i+1)*mini_batch_size]
    x = torch.from_numpy(data).to(cg.device)
    with torch.no_grad():
        f = net(x)
        # f = torch.round(f)
        # f = f.to(dtype=torch.int16)
        r.append(f.to('cpu'))

# %%
outputs = torch.squeeze(torch.cat(r))


# %%
if not cg.as_gray:
    # Test on a sample image
    test_image_path = './test_images/German_BMW_Police_Car_in_Munich_small.jpg'
    img = Image.open(test_image_path).convert('RGB')
    img = np.array(img)
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32)
    img = img / 255

    input = img[None, :]
    input = torch.from_numpy(input).float().to(cg.device)
    with torch.no_grad():
        output = net(input)
    output = output.cpu().detach().numpy()

    # Generate PCA visualization of feature maps
    pca_img = plot_utils.plot_pca_feature_map(
        output, 
        shape=(output.shape[2], output.shape[3]),
        out_path=plot_utils.PLOTS_FOLDER + 'feature_maps_pca.png'
    )


# %%
# Pick some random image patches and find some nearest image patches in the test dataset
fig, ax = plt.subplots()
plot_utils.plot_outputs_hist(outputs, ax)
fig.tight_layout()  # add this before saving the figure
fig.savefig(plot_utils.PLOTS_FOLDER + 'outputs_hist.png')

fig, ax = plt.subplots()
plot_utils.plot_outputs_dist(outputs, ax)
fig.tight_layout()  # add this before saving the figure
fig.savefig(plot_utils.PLOTS_FOLDER + 'outputs_dist.png')

plot_utils.plot_samples_vs_activenodes(outputs)

plot_utils.plot_receptive_fields(img_patches, outputs)

for i in range(5):
    plot_utils.plot_similar_patches(i, img_patches, outputs)

# Get the number of unique activation patterns first to avoid IndexError
_, _, num_unique_act_pattern = plot_utils.get_unique_pattern_and_counts(outputs, return_top_n=None)
safe_top_n = min(20000, num_unique_act_pattern-1)
ur, counts, num_unique_act_pattern = plot_utils.get_unique_pattern_and_counts(outputs, return_top_n=safe_top_n)

sample_inds_for_pattern_ind, pattern_inds_with_n_active_nodes = plot_utils.gen_sample_and_pattern_inds(outputs, ur, cg.device)

# Get available numbers of active nodes
available_nodes = sorted(pattern_inds_with_n_active_nodes.keys())
print(f"Available numbers of active nodes: {available_nodes}")

# Get smallest, median and largest
smallest = available_nodes[0]
largest = available_nodes[-1]
median = available_nodes[len(available_nodes)//2]

print(f"Plotting for nodes: smallest={smallest}, median={median}, largest={largest}")

# Plot only these three cases
for n_nodes in [smallest, median, largest]:
    plot_utils.plot_patches_with_n_active_nodes(
        pattern_inds_with_n_active_nodes, 
        sample_inds_for_pattern_ind, 
        img_patches, 
        n_nodes, 
        10, 
        10
    )
