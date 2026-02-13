_base_ = './rotated_rtmdet_l-3x-dota.py'

# checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
backbone=dict(
        type='fasternet_s',
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='/root/autodl-tmp/mmrotate-1.x/jcd/fasternet_s-epoch.299-val_acc1.81.2840.pth',
        )),
neck=dict(
        type='CSPNeXtPAFPN',
        CSPBlock_kernel_size=5,
        in_channels=[ 256, 512, 1024],
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
train_dataloader = dict(batch_size=8, num_workers=8)
work_dir = './work_dirs/rtmdet-r2/dota/only_fasternet-3x/'
test_evaluator = dict(_delete_=True, outfile_prefix=work_dir + 'only_faternet_Task1')
max_epochs = 3 * 12  # 36 epochs
base_lr = 0.004 / 16
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, weight_decay=0.0001)
