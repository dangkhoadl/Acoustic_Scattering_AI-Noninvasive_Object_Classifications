from transformers import ASTFeatureExtractor, ASTForAudioClassification

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    feature_extractor = ASTFeatureExtractor(
        sampling_rate=48000,
        padding="max_length",
        return_tensors="pt")

    # Load model
    model = ASTForAudioClassification.from_pretrained(
        "src/AST/ast-model")

    # Load data
    sig, sr = torchaudio.load(
        "DATA/round6_14_6_2022/MAMI_48k_sweet30Hz_len5s/MAMI_48k_sweet30Hz_len5s_001.wav")
    sig = torch.mean(sig, dim=0)

    inputs = feature_extractor(sig,
        sampling_rate=sr,
        padding="max_length",
        return_tensors="pt")
    print(f"{inputs['input_values'].shape = }") # (1, 1024, 128)

    with torch.no_grad():
        logits = model(inputs['input_values']).logits

    print(f"{logits.shape = }") # (1, 527)
    # Predict
    predicted_class_ids = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_ids]
    print(f"{predicted_label = }")
