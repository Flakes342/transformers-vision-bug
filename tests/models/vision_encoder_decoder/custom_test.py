from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import torch
from PIL import Image
import numpy as np

model_name = "nlpconnect/vit-gpt2-image-captioning"
model = VisionEncoderDecoderModel.from_pretrained(model_name)

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)  # for images
tokenizer = AutoTokenizer.from_pretrained(model_name)  # for text

image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
caption = "a white image"

pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

labels = tokenizer(caption, return_tensors="pt").input_ids

pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
# decoder_input_ids = torch.full(labels.shape, pad_token_id, dtype=labels.dtype)
# decoder_input_ids[:, 1:] = labels[:, :-1]
# decoder_input_ids[:, 0] = model.config.decoder_start_token_id

outputs = model(
    pixel_values=pixel_values,
    decoder_input_ids=None,
    labels=labels
)

print("Loss:", outputs.loss.item())