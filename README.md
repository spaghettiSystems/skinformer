# SkinFormer: Robust Vision Transformer for Automatic Skin Disease Identification
Get the model weights [here](https://www.kaggle.com/datasets/phantasm34/skinformer)

Example usage
```python
from transformers import BeitFeatureExtractor, FlaxBeitForImageClassification
from PIL import Image
import requests

url = "https://upload.wikimedia.org/wikipedia/commons/4/40/SolarAcanthosis.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')

feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224-pt22k")
model = FlaxBeitForImageClassification.from_pretrained("DIRECTORY/OF/DOWNLOADED/MODEL")

inputs = feature_extractor(images=image, return_tensors="np")
outputs = model(**inputs, output_attentions=True)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
```