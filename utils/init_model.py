import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

def apply_sparse_init(module):
    """Apply orthogonal initialization to linear or convolution layers."""
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        nn.init.orthogonal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_model(mil, args):
    """Initialize the MIL model, loss function, optimizer, and scheduler based on args."""
    if args.model == 'dsmil':
        args.instance_nrom = False
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes, instance_norm=args.instance_nrom).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes,
                                       dropout_v=args.dropout_node, nonlinear=args.non_linearity,
                                       instance_norm=args.instance_nrom).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()

    elif args.model == 'abmil':
        milnet = mil.GatedAttention(args.feats_size, args.num_classes).cuda()

    elif args.model == 'HistoFL':
        milnet = mil.MIL_Attention_fc(n_classes=args.num_classes, feats_size=args.feats_size,
                                      batch_norm=args.batch_norm).cuda()

    elif args.model == 'frmil':
        args.n_heads = 8
        milnet = mil.FRMIL(args).cuda()

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # Define loss function
    criterion = nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCEWithLogitsLoss()

    # Optimizer and learning rate scheduler
    optimizer = Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=5e-6)

    return milnet, criterion, optimizer, scheduler


def save_model(args, fold, run, save_path, model, thresholds_optimal):
    """Save model weights and print save message."""
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.join(save_path, f'fold_{fold}_{run + 1}.pth')
    torch.save(model.state_dict(), save_name)
    print_save_message(args, save_name, thresholds_optimal)


def print_save_message(args, save_name, thresholds_optimal):
    """Print where the model was saved and its thresholds."""
    print(f'Best model saved at: {save_name}')
    
    print('Best thresholds:', thresholds_optimal)
