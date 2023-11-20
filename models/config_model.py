import torch
import torch.optim as optim

from models.image_extractor import get_image_extractor
from models.sffnet import STF
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def configure_model(args, dataset):
    image_extractor = None
    gfn = None

    model = STF(dataset, args)
    model = model.to(device)
    # configure optimizer
    model_params = []
    fusion_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'fusion' in name:
                fusion_params.append(param)
            else:
                model_params.append(param)
    # model_params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [{'params': model_params},
                    {'params': fusion_params, 'lr': args.fuse_lr, 'weight_decay': args.fuse_wd}]
    if not args.use_precomputed_features:
        image_extractor = get_image_extractor(arch=args.image_extractor, pretrained=True)
        image_extractor = image_extractor.to(device)
        if args.use_gfn:
            gfn = get_image_extractor(arch=args.gfn_arch, pretrained=True)
            gfn = gfn.to(device)
        if args.finetune_backbone:
            ie_parameters = [param for name, param in image_extractor.named_parameters()]
            optim_params.append({'params': ie_parameters, 'lr': args.lrg})
            gfn_parameters = [param for name, param in gfn.named_parameters()]
            optim_params.append({'params': gfn_parameters,'lr': args.lrg})
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)

    return image_extractor, gfn, model, optimizer

