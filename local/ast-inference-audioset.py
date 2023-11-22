from transformers import AutoFeatureExtractor, ASTForAudioClassification
from datasets import load_dataset
import torch


if __name__ == "__main__":
    # Load dset
    dataset = load_dataset(
        "hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    dataset = dataset.sort("id")
    sampling_rate = dataset.features["audio"].sampling_rate
    print(f"{sampling_rate = }") # 16000

    # Load model
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "src/AST/ast-processor")
    model = ASTForAudioClassification.from_pretrained(
        "src/AST/ast-model")

    # Input audio
    print(f"{dataset[0]['audio']['array'].shape = }")
        # (93680,)
    inputs = feature_extractor(
        dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    print(f"{inputs['input_values'].shape = }")
        # (1, 1024, 128)

    # Infer
    with torch.no_grad():
        logits = model(inputs['input_values']).logits
    print(f"{logits.shape = }") # (1, 527)

    # Predict
    predicted_class_ids = torch.argmax(logits, dim=-1).item()
    predicted_label = model.config.id2label[predicted_class_ids]
    print(f"{predicted_label = }")
