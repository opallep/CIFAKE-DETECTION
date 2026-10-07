import torch
import config
from models import get_model

def quick_test():
    print("=" * 50)
    print(" QUICK DRY-RUN TEST")
    print("=" * 50)
    
    device = config.DEVICE
    print(f"Device: {device}")
    
    # Test all 3 models
    for model_name in config.ALL_MODELS:
        print(f"\n[Testing] Model: {model_name}")
        try:
            model = get_model(model_name)
            model.eval()
            
            # Determine input size
            if model_name == config.MODEL_LIGHTWEIGHT:
                img_size = config.IMG_SIZE_LIGHTWEIGHT
            else:
                img_size = config.IMG_SIZE_TRANSFER
                
            batch_size = 2
            dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
                
            print(f"  Input shape : {dummy_input.shape}")
            print(f"  Output shape: {output.shape}")
            print(f"  Success! Forward pass completed.")
            
        except Exception as e:
            print(f"  [ERROR] {e}")

if __name__ == '__main__':
    quick_test()
