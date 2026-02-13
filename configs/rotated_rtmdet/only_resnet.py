_base_ = './rotated_rtmdet_l-3x-dota.py'

# checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=False),  # True
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
neck=dict(
        type='CSPNeXtPAFPN',
        CSPBlock_kernel_size=5,
        in_channels=[ 512, 1024, 2048],
        out_channels=96,
        num_csp_blocks=1,
    ),
    bbox_head=dict(
        in_channels=96,
        feat_channels=96,
        exp_on_reg=False,
        loss_bbox=dict(type='RotatedIoULoss', mode='linear', loss_weight=2.0),
    ))

# batch_size = (1 GPUs) x (8 samples per GPU) = 8

max_epochs = 3 * 12  # 36 epochs
base_lr = 0.004 / 16

train_dataloader = dict(batch_size=8, num_workers=8)
work_dir = './work_dirs/rtmdet-r2/dota/only_resnet-3x/'
test_evaluator = dict(outfile_prefix=work_dir + 'only_resnet_Task1')