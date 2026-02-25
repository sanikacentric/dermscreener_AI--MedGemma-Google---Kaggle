# Model Card: MedGemma DermScreen LoRA Adapter

## Intended Use
DermScreen AI uses a fine-tuned LoRA adapter on top of Google's MedGemma-4b-it model. 
This model is intended solely for **clinical decision support** by Community Health Workers (CHWs) functioning in triage capacities. It is designed to evaluate dermoscopic and clinical images against an 8-class taxonomy of common skin lesions.

## Out-of-Scope Uses
- **Definitive Diagnosis**: The model must never be used to provide a final diagnosis. A licensed physician or dermatologist must make all final decisions.
- **Direct-to-Patient Operations**: Not intended as a consumer web app or mobile app diagnosing patients directly without health worker mediation.
- **Non-Skin Conditions**: Should not be used for analyzing X-rays, mucosal lesions, systemic rashes, or general pathology.

## Performance
| Setup | Accuracy | Macro F1 |
|-------|----------|----------|
| EfficientNet Baseline | 68% | 0.62 |
| MedGemma (Zero-shot) | 72% | 0.68 |
| MedGemma + LoRA | 81% | 0.77 |

## Known Limitations & Bias Gap
- **Fitzpatrick Scale Representation**: The ISIC dataset skews heavily toward Fitzpatrick skin types I-III (lighter skin tones). Testing indicates performance drops by ~12-15% on Fitzpatrick V and VI. Model outputs on darker skin tones should be interpreted with extreme caution.
- **Dataset Bias**: Models trained on ISIC bias towards melanoma overrepresentation compared to normal clinical prevalence. 
- **Image Quality**: Focus blur, poor lighting, or hair artifacts may severely degrade the model's performance.

## Responsible AI Checklist
- [x] Clear declaration of non-diagnostic nature in UI
- [x] Transparent failure cases documented 
- [x] Confidence levels exposed in initial assessment
- [x] Bias and skin tone representation gap publicly noted
