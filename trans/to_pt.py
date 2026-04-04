import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unet_model import UNet


def load_checkpoint(checkpoint_path, n_channels=3, n_classes=1):
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=False)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def export_to_torchscript(model, output_path='unet.pt', input_shape=(1, 3, 572, 572)):
    model.eval()
    
    dummy_input = torch.randn(*input_shape)
    
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    print(f"Saved TorchScript model to {output_path}")
    
    return traced_model


def verify_model(model_path, input_shape=(1, 3, 572, 572)):
    model = torch.jit.load(model_path)
    model.eval()
    
    dummy_input = torch.randn(*input_shape)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    return output


if __name__ == '__main__':
    checkpoint_path = 'unet_carvana_scale1.0_epoch2.pth'
    output_path = 'unet.pt'
    
    print("Loading checkpoint...")
    model = load_checkpoint(checkpoint_path, n_classes=2)
    
    print("Exporting to TorchScript...")
    export_to_torchscript(model, output_path)
    
    print("Verifying exported model...")
    verify_model(output_path)
