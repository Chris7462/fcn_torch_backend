import torch
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import FCN_ResNet50_Weights
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('--width', type=int, default=1238, help='The width of the input image')
ap.add_argument('--height', type=int, default=374, help='The height of the input image')
ap.add_argument('--output-dir', type=str, help='The path to output .pt file')
ap.add_argument('--model', type=str, required=True, help='model_name must be "fcn_resnet50" or "fcn_resnet101"')
args = vars(ap.parse_args())
#args = {'height': 374, 'width': 1238, 'model': 'fcn_resnet50', 'output_dir': '../model'}

# 1. Load the pretrained model
model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT)
model.eval()

# 2. Create a wrapper that always returns only the main segmentation tensor
class FCNWrapper(torch.nn.Module):
    def __init__(self, fcn_model):
        super(FCNWrapper, self).__init__()
        self.fcn = fcn_model

    def forward(self, x):
        # FCN returns a dict with 'out' and 'aux' keys
        # We only want the main output tensor
        output = self.fcn(x)
        return output['out']  # Always return just the main segmentation tensor

# Wrap the model to ensure deterministic output
wrapped_model = FCNWrapper(model)
wrapped_model.eval()

# 3. Example input tensor (dummy input for tracing)
dummy_input = torch.randn(1, 3, args['height'], args['width'])

# 4. Export using TorchScript tracing
traced_script_module = torch.jit.trace(wrapped_model, dummy_input, strict=False)

# 5. Save the scripted model
traced_script_module.save(f"{args['output_dir']}/{args['model']}_{args['height']}x{args['width']}.pt")

print(f"TorchScript model saved to: {args['output_dir']}")
