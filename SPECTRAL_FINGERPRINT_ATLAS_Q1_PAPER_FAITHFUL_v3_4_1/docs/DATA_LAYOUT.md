# Expected data layout

The default scripts target Google Drive under `/content/drive/MyDrive`.

```text
MyDrive/
  Datasets/
    archive/generated_audio/                 # WaveFake generated audio folders
    LJSpeech-1.1/wavs/                       # LJSpeech bona fide
    jsut_ver1.1/basic5000/wav/               # JSUT bona fide
    LibriSeVoc/
      gt/
      wavenet/
      wavernn/
      melgan/
      parallel_wave_gan/
      wavegrad/
      diffwave/
  DOCTORAT/
    Mlaad/mlaad_v5/                          # MLAAD v5
    mailbs/                                  # local M-AILABS tree
  fingerprint_q1_outputs/                    # generated results; do not commit
```

Exact local directory names may differ. Edit only the CONFIG/root blocks in the pairing scripts when needed. Do not change scientific thresholds merely to make a dataset pass a guardrail.
