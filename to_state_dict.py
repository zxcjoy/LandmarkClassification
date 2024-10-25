import argparse

import torch

from model import LitModule


def save_params(ckpt_path, save_path):
    module = LitModule.load_from_checkpoint(ckpt_path)
    torch.save(module.model.state_dict(), save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pl to pytorch state dict")
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", default="checkpoints/effnet-b0.pth", type=str)

    args = parser.parse_args()

    save_params(args.checkpoint, args.output)
