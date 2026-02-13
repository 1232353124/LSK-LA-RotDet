_base_ = './rotated_rtmdet_l-3x-dota.py'

# checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        type='LWGANet',
        stem_dim=96,
        depths=(1, 4, 4, 2),
        att_kernel=(11, 11, 11, 11),
        norm_layer=dict(type='SyncBN', requires_grad=True),
        fork_feat=True,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained',
                      checkpoint="/root/autodl-tmp/mmrotate-1.x/jcd/lwganet_l2_e296.pth"),
        pretrained=None),
neck=dict(
        type='CSPNeXtPAFPN',
        CSPBlock_kernel_size=5,
        in_channels=[192, 384,768],
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
train_dataloader = dict(batch_size=4, num_workers=8)
work_dir = './work_dirs/rtmdet-r2/dota/only_lwganet-3x/'
max_epochs = 3 * 12  # 36 epochs
base_lr = 0.004 / 32

test_evaluator = dict(outfile_prefix=work_dir + 'only_lwganet_Task1')
