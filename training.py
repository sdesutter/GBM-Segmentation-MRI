import os
import time
import glob

import wandb

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import Compose, MapTransform, Activations, AsDiscrete

from monai.metrics import DiceMetric
from monai.networks.nets import DynUNet, SwinUNETR
from monai import data
from monai.data import decollate_batch
from monai.utils import set_determinism

import torch


###################################################
####################ENVIRONMENT####################
###################################################

set_determinism(seed=0)

num_CPU_cores = 16
num_workers_train = round(0.7*num_CPU_cores)
num_workers_val = num_CPU_cores - num_workers_train
num_init_workers_train = round(num_workers_train/2)
num_replace_workers_train = num_workers_train - num_init_workers_train
num_init_workers_val = round(num_workers_val/2)
num_replace_workers_val = num_workers_val - num_init_workers_val
VAL_AMP=True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




###################################################
####################PARAMETERS#####################
###################################################

arch = 'dyn' #'dyn' for DynUnet, 'swin' for SwinUNETR
modalities = ['t1', 't2', 't1ce', 'flair']

data_root = 'path/to/data/folder'
model_root = 'path/to/model/folder'

patch_size = (128,128,128)
batch_size = 4
max_epochs = 1000
val_interval = 10
lr = 1e-4
norm = 'INSTANCE'
if arch == 'dyn':
    dropout_rate = 0.2
elif arch == 'swin':
    dropout_rate = 0.5



###################################################
#####################FUNCTIONS#####################
###################################################

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    def __call__(self, data):
        # 1 = necrosis
        # 2 = edema
        # 4 = ET
        
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 4 to construct TC
            result.append(torch.logical_or(d[key] == 1, d[key] == 4))
            # merge labels 1, 2 and 4 to construct WT
            result.append(
                torch.logical_or(
                    torch.logical_or(d[key] == 2, d[key] == 4), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 4)
            d[key] = torch.stack(result, axis=0).float()
        return d
    
    
def get_data(root, train_val_or_test):
   folder = os.path.join(root, train_val_or_test)
   data_list = []
   for f in glob.glob(folder+'/*'):
       dict = {}
       dict['image'] = [os.path.join(f,f'{modality}.nii.gz') for modality in modalities]
       dict['label'] = os.path.join(f,'seg.nii.gz')
       data_list.append(dict)
   return data_list



###################################################
####################DATALOADER#####################
###################################################

# GET FILENAMES
train_files = get_data(data_root, 'Training')
val_files = get_data(data_root, 'Validation')


# GET TRANSFORMS
train_transforms = transforms.Compose(
    [
        # DETERMINISTIC TRANSFORMATIONS
        transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        
        # STOCHASTIC TRANSFORMATIONS (AUGMENTATION)
        transforms.RandCropByPosNegLabeld(keys=["image", "label"], spatial_size=patch_size, label_key="label", num_samples=2),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.9),
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.9)
        
    ]
)


val_transforms = transforms.Compose(
    [
        # DETERMINISTIC TRANSFORMATIONS
        transforms.LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest"))

    ]
)


# DATASET AND DATALOADER
train_ds = data.SmartCacheDataset(train_files, replace_rate=0.2, cache_num=200, transform=train_transforms, num_init_workers=num_init_workers_train, num_replace_workers=num_replace_workers_train)
val_ds = data.SmartCacheDataset(val_files, replace_rate=0.2, cache_num=100, transform=val_transforms, num_init_workers=num_init_workers_val, num_replace_workers=num_replace_workers_val)

train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers_train)
val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers_val)



###################################################
#################MODEL DEFINITION##################
###################################################

if arch == 'dyn':
    
    sizes = patch_size
    spacings = (1.0, 1.0, 1.0)
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {patch_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)
    
    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    
    model = DynUNet(
        spatial_dims=3,
        in_channels=len(modalities),
        out_channels=3,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        deep_supervision=False,
        norm_name=norm,
        dropout=dropout_rate
    ).to(device)
    
elif arch == 'swin':
    
    model = SwinUNETR(
        img_size=patch_size,
        in_channels=len(modalities),
        out_channels=3,
        feature_size=48,
        drop_rate=dropout_rate,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)


loss_function = DiceCELoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=1e-5)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


def inference(input):
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=patch_size,
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)

scaler = torch.cuda.amp.GradScaler()
torch.backends.cudnn.benchmark = True



###################################################
#####################TRAINING######################
###################################################

best_metric = -1
best_metric_epoch = -1
best_metrics_epochs_and_time = [[], [], []]
epoch_loss_values = []
metric_values = []
metric_values_tc = []
metric_values_wt = []
metric_values_et = []

total_start = time.time()
for epoch in range(max_epochs):
    epoch_start = time.time()
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step_start = time.time()
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}"
            f", train_loss: {loss.item():.4f}"
            f", step time: {(time.time() - step_start):.4f}"
        )
    lr_scheduler.step()
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():

            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                val_outputs = inference(val_inputs)
                val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                dice_metric(y_pred=val_outputs, y=val_labels)
                dice_metric_batch(y_pred=val_outputs, y=val_labels)

            metric = dice_metric.aggregate().item()
            metric_values.append(metric)
            metric_batch = dice_metric_batch.aggregate()
            metric_tc = metric_batch[0].item()
            metric_values_tc.append(metric_tc)
            metric_wt = metric_batch[1].item()
            metric_values_wt.append(metric_wt)
            metric_et = metric_batch[2].item()
            metric_values_et.append(metric_et)
            dice_metric.reset()
            dice_metric_batch.reset()

            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                best_metrics_epochs_and_time[0].append(best_metric)
                best_metrics_epochs_and_time[1].append(best_metric_epoch)
                best_metrics_epochs_and_time[2].append(time.time() - total_start)
                torch.save(
                    model.state_dict(),
                    os.path.join(model_root, f"seg_{arch}_{'_'.join(modalities)}.pth"),
                )
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                f"\nbest mean dice: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
    print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
total_time = time.time() - total_start

wandb.log({"best_metric": best_metric, "best_epoch": best_metric_epoch, "total_time": total_time})

