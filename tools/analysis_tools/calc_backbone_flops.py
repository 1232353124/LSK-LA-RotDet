import torch
from mmengine.config import Config
from mmengine.registry import MODELS
from thop import profile

def main():
    cfg_path = "configs/rotated_rtmdet/only_lsknet.py"
    cfg = Config.fromfile(cfg_path)

    # backbone
    backbone_cfg = cfg.model.backbone
    backbone = MODELS.build(backbone_cfg)

    # dummy input
    input_res = (1, 3, 1024, 1024)
    x = torch.randn(input_res).cuda()   # 放到GPU
    backbone = backbone.cuda()          # 放到GPU

    flops, params = profile(backbone, inputs=(x,), verbose=False)

    print("="*60)
    print(f"Backbone: {backbone_cfg['type']}")
    print(f"Input resolution: {input_res}")
    print(f"FLOPs: {flops/1e9:.2f} G")
    print(f"Params: {params/1e6:.2f} M")
    print("="*60)

if __name__ == "__main__":
    main()
