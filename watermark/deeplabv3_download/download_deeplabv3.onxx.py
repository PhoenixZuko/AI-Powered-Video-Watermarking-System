import torch
import torchvision.models as models

# Descărcăm modelul DeepLabV3 pre-antrenat
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval()

# Creăm un input tensor dummy (imaginea trebuie să aibă dimensiunea 224x224)
dummy_input = torch.randn(1, 3, 224, 224)

# Exportăm modelul în format ONNX
torch.onnx.export(model, dummy_input, "deeplabv3.onnx", opset_version=11, input_names=["input"], output_names=["output"])

print("✅ Modelul DeepLabV3 a fost salvat ca ONNX!")
