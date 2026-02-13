_base_ = './rotated_rtmdet_l-3x-dota.py'

# checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'  # noqa

model = dict(
    backbone=dict(
        type='DecoupleNet',
        in_chans=3,
        embed_dim=32,
        depths=(1, 6, 6, 2),
        att_kernel=(9, 9, 9, 9),
        drop_path_rate=0.1,
        fork_feat=True,
        init_cfg=dict(type='Pretrained',
                      checkpoint="/root/autodl-tmp/mmrotate-1.x/jcd/DecoupleNet_D0.pth"),
        pretrained=None),
neck=dict(
        type='CSPNeXtPAFPN',
        CSPBlock_kernel_size=5,
        in_channels=[64, 128, 256],
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
work_dir = './work_dirs/rtmdet-r2/dota/only_decouplenet-3x/'
test_evaluator = dict(_delete_=True, outfile_prefix=work_dir + 'only_decouplenet_Task1')
