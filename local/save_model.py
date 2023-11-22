#!/usr/bin/env python3
from transformers import ASTFeatureExtractor, ASTModel, \
    Wav2Vec2FeatureExtractor, HubertForSequenceClassification, \
    AutoFeatureExtractor, Wav2Vec2ConformerForSequenceClassification

#### AST
# feature_extractor = ASTFeatureExtractor \
#     .from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
# feature_extractor.save_pretrained("src/AST/ast-feature_extractor")

model = ASTModel \
    .from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model.save_pretrained("src/AST/ast-model")

#### Hubert
# feature_extractor = Wav2Vec2FeatureExtractor \
#     .from_pretrained("facebook/hubert-base-ls960")
# feature_extractor.save_pretrained("src/Hubert/hubert-feature-extractor")

model_1 = HubertForSequenceClassification \
    .from_pretrained("facebook/hubert-base-ls960")
model_1.save_pretrained("src/Hubert/hubert-model")

model_2 = HubertForSequenceClassification \
    .from_pretrained("superb/hubert-base-superb-ks")
model_2.save_pretrained("src/Hubert/hubert-speechcommand-model")


#### wav2vec- Conformer
# feature_extractor = AutoFeatureExtractor \
#     .from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
# feature_extractor.save_pretrained("src/wav2vec2-Conformer/wav2vec2_conformer-feature_extractor")

model = Wav2Vec2ConformerForSequenceClassification \
    .from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
model.save_pretrained("src/wav2vec2-Conformer/wav2vec2_conformer-model")
