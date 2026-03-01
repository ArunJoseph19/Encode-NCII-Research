# Data Directory

## Structure

```
data/
├── consenting_subjects/      ← Face images for LoRA training
│   └── subject_001/          ← 15-30 images of one person
├── celeba_hq/                ← Downloaded CelebA-HQ subset (via download_faces.py)
└── safety_prompts/           ← Text prompts for POC 2 (deferred)
```

## Data Collection Ethics

- **No real NCII** is created, handled, or stored at any point in this research.
- For POC 1, we use publicly available face datasets (CelebA-HQ) or images from consenting research participants.
- All subject data is stored locally and not committed to version control.
- See `CONSENT_TEMPLATE.md` for the informed consent protocol if using non-public subjects.

## Download

Run `python data/download_faces.py` to download a CelebA-HQ subset for a single identity.

## Folder Convention

Each subject gets their own folder: `consenting_subjects/subject_NNN/`
Images should be:
- High resolution (512×512 minimum, 1024×1024 preferred)
- Clear face visibility, varied angles/lighting
- 15-30 images per subject for reliable LoRA training
