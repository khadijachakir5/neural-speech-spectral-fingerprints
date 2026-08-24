#!/usr/bin/env python3


# Purpose: Assign the frozen MLAAD taxonomy and construct the STRICT and RELAXED confirmatory populations.

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PHASE0_DIR = Path(
    "/content/drive/MyDrive/fingerprint_q1_outputs/phase0_mlaad_v2_2_canonical"
)


PHASE0_ZIP = Path("")

OUTPUT_DIR = Path(
    "/content/drive/MyDrive/fingerprint_q1_outputs/phase0b/"
    "phase0b_mlaad_taxonomy_v1_2_new_story"
)

TAXONOMY_PATH = Path(
    "/content/drive/MyDrive/fingerprint_q1_outputs/taxonomy/"
    "mlaad_generator_taxonomy_v1.csv"
)

EXPECTED_TOTAL_PAIRS = 84_000
EXPECTED_GENERATORS = 55
EXPECTED_LANGUAGES = 8
MIN_GENERATORS_PER_FAMILY = 3
MIN_PAIRS_PER_GENERATOR = 30
CONFIRMATORY_CONFIDENCE = {"high", "medium"}
BUILD_RELAXED_PROTOCOL = True
OVERWRITE_TAXONOMY_CSV = True
CREATE_BACKUP_BEFORE_OVERWRITE = True

VERSION = "MLAAD-PHASE0B-v1.2-NEW-STORY"
RUNTIME_DIR = Path("/content/phase0b_mlaad_taxonomy_v1_2_new_story_runtime")


EMBEDDED_TAXONOMY: List[Dict[str, str]] = [
  {
    "independent_generator_id": "Mars5",
    "pipeline_type": "codec-token TTS (AR + NAR refinement)",
    "acoustic_model": "AR Transformer + multinomial DDPM",
    "waveform_architecture": "Vocos decoder over EnCodec codebooks",
    "waveform_family": "GAN vocoder",
    "representation": "EnCodec RVQ tokens",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/Camb-ai/MARS5-TTS | https://github.com/gemelo-ai/vocos | https://github.com/facebookresearch/encodec",
    "taxonomy_notes": "The DDPM refines codec codebooks; final waveform synthesis is performed by Vocos. Family is assigned from the waveform stage, not from the token-refinement stage."
  },
  {
    "independent_generator_id": "MatchaTTS",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Matcha-TTS conditional flow matching",
    "waveform_architecture": "HiFi-GAN",
    "waveform_family": "GAN vocoder",
    "representation": "log-mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/shivammehta25/Matcha-TTS | https://github.com/jik876/hifi-gan",
    "taxonomy_notes": "Official Matcha implementation synthesizes mel spectrograms and uses an off-the-shelf HiFi-GAN vocoder."
  },
  {
    "independent_generator_id": "MeloTTS",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "MeloTTS (VITS/VITS2-based)",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "multilingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/myshell-ai/MeloTTS | https://github.com/jaywalnut310/vits",
    "taxonomy_notes": "MeloTTS states that its implementation is based on VITS, VITS2 and Bert-VITS2."
  },
  {
    "independent_generator_id": "Metavoice-1B",
    "pipeline_type": "codec-token TTS",
    "acoustic_model": "causal GPT + NAR Transformer over EnCodec tokens",
    "waveform_architecture": "multi-band waveform diffusion + DeepFilterNet",
    "waveform_family": "Diffusion waveform",
    "representation": "EnCodec RVQ tokens",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/metavoiceio/metavoice-src | https://github.com/facebookresearch/encodec",
    "taxonomy_notes": "Official architecture predicts EnCodec tokens, then generates waveform with multi-band diffusion and cleans it with DeepFilterNet."
  },
  {
    "independent_generator_id": "OpenVoiceV2",
    "pipeline_type": "base TTS + tone-color voice conversion",
    "acoustic_model": "MeloTTS base model + OpenVoice tone-color converter",
    "waveform_architecture": "VITS/MeloTTS generator followed by tone-color converter",
    "waveform_family": "Hybrid TTS + voice conversion",
    "representation": "mel/latent speech representation + speaker embedding",
    "training_language_scope": "multilingual",
    "taxonomy_confidence": "medium",
    "taxonomy_source": "https://github.com/myshell-ai/OpenVoice | https://github.com/myshell-ai/MeloTTS",
    "taxonomy_notes": "OpenVoice V2 is a hybrid chain. The exact base checkpoint may vary; the family is therefore kept separate from plain VITS."
  },
  {
    "independent_generator_id": "WhisperSpeech",
    "pipeline_type": "two-stage token-based TTS",
    "acoustic_model": "Whisper semantic model + EnCodec acoustic model",
    "waveform_architecture": "Vocos decoder",
    "waveform_family": "GAN vocoder",
    "representation": "Whisper semantic tokens + EnCodec acoustic tokens",
    "training_language_scope": "multilingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/WhisperSpeech/WhisperSpeech | https://github.com/gemelo-ai/vocos | https://github.com/facebookresearch/encodec",
    "taxonomy_notes": "Official architecture lists Whisper semantic tokens, EnCodec acoustic tokens and Vocos as vocoder."
  },
  {
    "independent_generator_id": "e2-tts",
    "pipeline_type": "two-stage flow-matching TTS + vocoder",
    "acoustic_model": "E2-TTS flow-matching Transformer",
    "waveform_architecture": "Vocos (F5-TTS implementation)",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "medium",
    "taxonomy_source": "https://arxiv.org/abs/2406.18009 | https://github.com/SWivid/F5-TTS | https://github.com/gemelo-ai/vocos",
    "taxonomy_notes": "The E2-TTS paper specifies a mel generator plus a vocoder; the MLAAD implementation is associated with the official F5-TTS codebase, which uses Vocos."
  },
  {
    "independent_generator_id": "f5-tts",
    "pipeline_type": "two-stage flow-matching TTS + vocoder",
    "acoustic_model": "F5-TTS Diffusion Transformer + ConvNeXt V2",
    "waveform_architecture": "Vocos",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/SWivid/F5-TTS | https://github.com/gemelo-ai/vocos",
    "taxonomy_notes": "Official F5-TTS implementation generates mel spectrograms and reconstructs waveform with Vocos."
  },
  {
    "independent_generator_id": "facebook_mms-tts-deu",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "MMS-TTS / VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://huggingface.co/facebook/mms-tts | https://github.com/jaywalnut310/vits",
    "taxonomy_notes": "Each MMS-TTS checkpoint is language-specific and exposed through the VITS text-to-waveform architecture."
  },
  {
    "independent_generator_id": "facebook_mms-tts-eng",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "MMS-TTS / VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://huggingface.co/facebook/mms-tts | https://github.com/jaywalnut310/vits",
    "taxonomy_notes": "Each MMS-TTS checkpoint is language-specific and exposed through the VITS text-to-waveform architecture."
  },
  {
    "independent_generator_id": "facebook_mms-tts-fra",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "MMS-TTS / VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://huggingface.co/facebook/mms-tts | https://github.com/jaywalnut310/vits",
    "taxonomy_notes": "Each MMS-TTS checkpoint is language-specific and exposed through the VITS text-to-waveform architecture."
  },
  {
    "independent_generator_id": "facebook_mms-tts-rus",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "MMS-TTS / VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://huggingface.co/facebook/mms-tts | https://github.com/jaywalnut310/vits",
    "taxonomy_notes": "Each MMS-TTS checkpoint is language-specific and exposed through the VITS text-to-waveform architecture."
  },
  {
    "independent_generator_id": "facebook_mms-tts-ukr",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "MMS-TTS / VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://huggingface.co/facebook/mms-tts | https://github.com/jaywalnut310/vits",
    "taxonomy_notes": "Each MMS-TTS checkpoint is language-specific and exposed through the VITS text-to-waveform architecture."
  },
  {
    "independent_generator_id": "griffin_lim",
    "pipeline_type": "signal-processing baseline",
    "acoustic_model": "none (spectrogram inversion baseline)",
    "waveform_architecture": "Griffin-Lim iterative phase rebuild",
    "waveform_family": "Classical phase rebuild",
    "representation": "magnitude or mel-derived spectrogram",
    "training_language_scope": "language-agnostic",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://doi.org/10.1109/TASSP.1984.1164317",
    "taxonomy_notes": "Non-neural baseline; waveform is reconstructed iteratively from a magnitude spectrogram."
  },
  {
    "independent_generator_id": "microsoft_speecht5_tts",
    "pipeline_type": "two-stage encoder-decoder TTS + vocoder",
    "acoustic_model": "SpeechT5 Transformer encoder-decoder",
    "waveform_architecture": "HiFi-GAN",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/microsoft/SpeechT5 | https://huggingface.co/microsoft/speecht5_hifigan | https://github.com/jik876/hifi-gan",
    "taxonomy_notes": "The standard SpeechT5 TTS pipeline uses the dedicated SpeechT5 HiFi-GAN vocoder checkpoint."
  },
  {
    "independent_generator_id": "optispeech",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "OptiSpeech ConvNeXt/JETS-style acoustic model",
    "waveform_architecture": "WaveNeXt waveform generator",
    "waveform_family": "GAN vocoder",
    "representation": "phoneme-conditioned hidden features to waveform",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "medium",
    "taxonomy_source": "https://github.com/mush42/optispeech",
    "taxonomy_notes": "The repository identifies a WaveNeXt generator and discriminator. Confidence is medium because the project is marked work-in-progress and checkpoint configuration may evolve."
  },
  {
    "independent_generator_id": "parler_tts_large_v1",
    "pipeline_type": "codec language-model TTS",
    "acoustic_model": "Parler-TTS Large v1 autoregressive decoder",
    "waveform_architecture": "Descript Audio Codec (DAC) decoder",
    "waveform_family": "Neural codec decoder",
    "representation": "DAC RVQ audio tokens",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/huggingface/parler-tts | https://github.com/descriptinc/descript-audio-codec",
    "taxonomy_notes": "Parler-TTS uses a MusicGen-like autoregressive decoder and DAC to recover waveform from generated audio codes."
  },
  {
    "independent_generator_id": "parler_tts_mini_v0.1",
    "pipeline_type": "codec language-model TTS",
    "acoustic_model": "Parler-TTS Mini v0.1 autoregressive decoder",
    "waveform_architecture": "Descript Audio Codec (DAC) decoder",
    "waveform_family": "Neural codec decoder",
    "representation": "DAC RVQ audio tokens",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/huggingface/parler-tts | https://github.com/descriptinc/descript-audio-codec",
    "taxonomy_notes": "Parler-TTS uses a MusicGen-like autoregressive decoder and DAC to recover waveform from generated audio codes."
  },
  {
    "independent_generator_id": "parler_tts_mini_v1",
    "pipeline_type": "codec language-model TTS",
    "acoustic_model": "Parler-TTS Mini v1 autoregressive decoder",
    "waveform_architecture": "Descript Audio Codec (DAC) decoder",
    "waveform_family": "Neural codec decoder",
    "representation": "DAC RVQ audio tokens",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/huggingface/parler-tts | https://github.com/descriptinc/descript-audio-codec",
    "taxonomy_notes": "Parler-TTS uses a MusicGen-like autoregressive decoder and DAC to recover waveform from generated audio codes."
  },
  {
    "independent_generator_id": "suno_bark",
    "pipeline_type": "codec language-model text-to-audio",
    "acoustic_model": "Bark semantic/coarse/fine GPT stack",
    "waveform_architecture": "EnCodec decoder",
    "waveform_family": "Neural codec decoder",
    "representation": "EnCodec RVQ audio tokens",
    "training_language_scope": "multilingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/suno-ai/bark | https://github.com/facebookresearch/encodec",
    "taxonomy_notes": "Bark uses GPT-style models over a quantized EnCodec audio representation; small and full variants share the waveform-decoder family."
  },
  {
    "independent_generator_id": "suno_bark-small",
    "pipeline_type": "codec language-model text-to-audio",
    "acoustic_model": "Bark semantic/coarse/fine GPT stack",
    "waveform_architecture": "EnCodec decoder",
    "waveform_family": "Neural codec decoder",
    "representation": "EnCodec RVQ audio tokens",
    "training_language_scope": "multilingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/suno-ai/bark | https://github.com/facebookresearch/encodec",
    "taxonomy_notes": "Bark uses GPT-style models over a quantized EnCodec audio representation; small and full variants share the waveform-decoder family."
  },
  {
    "independent_generator_id": "tts_models_de_css10_vits-neon",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "VITS-neon",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Registry identifies a VITS-neon model with no separate external vocoder."
  },
  {
    "independent_generator_id": "tts_models_de_thorsten_tacotron2-DCA",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Tacotron2-DCA",
    "waveform_architecture": "Fullband-MelGAN",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: Fullband-MelGAN."
  },
  {
    "independent_generator_id": "tts_models_de_thorsten_tacotron2-DDC",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Tacotron2-DDC",
    "waveform_architecture": "HiFi-GAN v1",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: HiFi-GAN v1."
  },
  {
    "independent_generator_id": "tts_models_de_thorsten_vits",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "VITS produces waveform internally; no separate vocoder."
  },
  {
    "independent_generator_id": "tts_models_en_blizzard2013_capacitron-t2-c50",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Capacitron Tacotron2 c50",
    "waveform_architecture": "HiFi-GAN v2",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: HiFi-GAN v2."
  },
  {
    "independent_generator_id": "tts_models_en_ek1_tacotron2",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Tacotron2",
    "waveform_architecture": "WaveGrad",
    "waveform_family": "Diffusion waveform",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: WaveGrad."
  },
  {
    "independent_generator_id": "tts_models_en_jenny_jenny",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "Jenny (VITS)",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry identifies Jenny as a VITS model."
  },
  {
    "independent_generator_id": "tts_models_en_ljspeech_fast_pitch",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "FastPitch",
    "waveform_architecture": "HiFi-GAN v2",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: HiFi-GAN v2."
  },
  {
    "independent_generator_id": "tts_models_en_ljspeech_glow-tts",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Glow-TTS",
    "waveform_architecture": "Multiband-MelGAN",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: Multiband-MelGAN."
  },
  {
    "independent_generator_id": "tts_models_en_ljspeech_neural_hmm",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Neural HMM TTS",
    "waveform_architecture": "HiFi-GAN v2",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: HiFi-GAN v2."
  },
  {
    "independent_generator_id": "tts_models_en_ljspeech_overflow",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Overflow",
    "waveform_architecture": "HiFi-GAN v2",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: HiFi-GAN v2."
  },
  {
    "independent_generator_id": "tts_models_en_ljspeech_speedy-speech",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "SpeedySpeech",
    "waveform_architecture": "HiFi-GAN v2",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: HiFi-GAN v2."
  },
  {
    "independent_generator_id": "tts_models_en_ljspeech_tacotron2-DCA",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Tacotron2-DCA",
    "waveform_architecture": "Multiband-MelGAN",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: Multiband-MelGAN."
  },
  {
    "independent_generator_id": "tts_models_en_ljspeech_tacotron2-DDC",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Tacotron2-DDC",
    "waveform_architecture": "HiFi-GAN v2",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: HiFi-GAN v2."
  },
  {
    "independent_generator_id": "tts_models_en_ljspeech_tacotron2-DDC_ph",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Tacotron2-DDC (phoneme input)",
    "waveform_architecture": "UnivNet",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: UnivNet."
  },
  {
    "independent_generator_id": "tts_models_en_ljspeech_vits",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "VITS produces waveform internally."
  },
  {
    "independent_generator_id": "tts_models_en_ljspeech_vits--neon",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "VITS-neon",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "VITS-neon produces waveform internally."
  },
  {
    "independent_generator_id": "tts_models_en_multi-dataset_tortoise-v2",
    "pipeline_type": "hybrid autoregressive + diffusion TTS",
    "acoustic_model": "Tortoise autoregressive decoder + diffusion decoder",
    "waveform_architecture": "UnivNet",
    "waveform_family": "GAN vocoder",
    "representation": "discrete acoustic codes + mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/neonbjb/tortoise-tts",
    "taxonomy_notes": "Official Tortoise repository states it uses an autoregressive decoder, a diffusion decoder and UnivNet vocoder."
  },
  {
    "independent_generator_id": "tts_models_en_sam_tacotron-DDC",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Tacotron-DDC",
    "waveform_architecture": "HiFi-GAN v2",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: HiFi-GAN v2."
  },
  {
    "independent_generator_id": "tts_models_es_css10_vits",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "VITS produces waveform internally."
  },
  {
    "independent_generator_id": "tts_models_es_mai_tacotron2-DDC",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Tacotron2-DDC",
    "waveform_architecture": "Fullband-MelGAN (universal)",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: universal Fullband-MelGAN."
  },
  {
    "independent_generator_id": "tts_models_fr_css10_vits",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "VITS produces waveform internally."
  },
  {
    "independent_generator_id": "tts_models_fr_mai_tacotron2-DDC",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Tacotron2-DDC",
    "waveform_architecture": "Fullband-MelGAN (universal)",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: universal Fullband-MelGAN."
  },
  {
    "independent_generator_id": "tts_models_it_mai_female_glow-tts",
    "pipeline_type": "two-stage acoustic model + phase rebuild",
    "acoustic_model": "Glow-TTS",
    "waveform_architecture": "Griffin-Lim fallback",
    "waveform_family": "Classical phase rebuild",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "medium",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "The Coqui registry has no default neural vocoder for this checkpoint; the standard fallback is Griffin-Lim."
  },
  {
    "independent_generator_id": "tts_models_it_mai_female_vits",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "VITS produces waveform internally."
  },
  {
    "independent_generator_id": "tts_models_it_mai_male_glow-tts",
    "pipeline_type": "two-stage acoustic model + phase rebuild",
    "acoustic_model": "Glow-TTS",
    "waveform_architecture": "Griffin-Lim fallback",
    "waveform_family": "Classical phase rebuild",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "medium",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "The Coqui registry has no default neural vocoder for this checkpoint; the standard fallback is Griffin-Lim."
  },
  {
    "independent_generator_id": "tts_models_it_mai_male_vits",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "VITS produces waveform internally."
  },
  {
    "independent_generator_id": "tts_models_multilingual_multi-dataset_bark",
    "pipeline_type": "codec language-model text-to-audio",
    "acoustic_model": "Bark semantic/coarse/fine GPT stack",
    "waveform_architecture": "EnCodec decoder",
    "waveform_family": "Neural codec decoder",
    "representation": "EnCodec RVQ audio tokens",
    "training_language_scope": "multilingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/suno-ai/bark | https://github.com/facebookresearch/encodec",
    "taxonomy_notes": "Coqui wrapper of Bark; waveform is reconstructed by EnCodec."
  },
  {
    "independent_generator_id": "tts_models_multilingual_multi-dataset_xtts_v1.1",
    "pipeline_type": "codec-token voice-cloning TTS",
    "acoustic_model": "XTTS v1.1 GPT acoustic model",
    "waveform_architecture": "HifiDecoder (HiFi-GAN-derived)",
    "waveform_family": "GAN vocoder",
    "representation": "discrete acoustic tokens / GPT latents",
    "training_language_scope": "multilingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/models/xtts.py",
    "taxonomy_notes": "Official XTTS implementation imports and uses HifiDecoder."
  },
  {
    "independent_generator_id": "tts_models_multilingual_multi-dataset_xtts_v2",
    "pipeline_type": "codec-token voice-cloning TTS",
    "acoustic_model": "XTTS v2 GPT acoustic model",
    "waveform_architecture": "HifiDecoder (HiFi-GAN-derived)",
    "waveform_family": "GAN vocoder",
    "representation": "discrete acoustic tokens / GPT latents",
    "training_language_scope": "multilingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/models/xtts.py",
    "taxonomy_notes": "Official XTTS implementation imports and uses HifiDecoder."
  },
  {
    "independent_generator_id": "tts_models_pl_mai_female_vits",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "VITS produces waveform internally."
  },
  {
    "independent_generator_id": "tts_models_uk_mai_glow-tts",
    "pipeline_type": "two-stage acoustic model + vocoder",
    "acoustic_model": "Glow-TTS",
    "waveform_architecture": "Multiband-MelGAN",
    "waveform_family": "GAN vocoder",
    "representation": "mel spectrogram",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "Coqui registry default vocoder: Multiband-MelGAN."
  },
  {
    "independent_generator_id": "tts_models_uk_mai_vits",
    "pipeline_type": "end-to-end neural TTS",
    "acoustic_model": "VITS",
    "waveform_architecture": "integrated VITS waveform decoder",
    "waveform_family": "Integrated GAN (VITS)",
    "representation": "latent / spectrogram representation",
    "training_language_scope": "monolingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://github.com/coqui-ai/TTS/blob/dev/TTS/.models.json | https://github.com/coqui-ai/TTS",
    "taxonomy_notes": "VITS produces waveform internally."
  },
  {
    "independent_generator_id": "vixTTS",
    "pipeline_type": "codec-token voice-cloning TTS",
    "acoustic_model": "viXTTS (fine-tuned XTTS v2.0.3)",
    "waveform_architecture": "HifiDecoder (HiFi-GAN-derived)",
    "waveform_family": "GAN vocoder",
    "representation": "discrete acoustic tokens / GPT latents",
    "training_language_scope": "multilingual",
    "taxonomy_confidence": "high",
    "taxonomy_source": "https://huggingface.co/capleaf/viXTTS | https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/models/xtts.py",
    "taxonomy_notes": "The model card states viXTTS is fine-tuned from XTTS-v2.0.3; waveform architecture is inherited from XTTS."
  }
]

TAXONOMY_FIELDS = [
    "pipeline_type",
    "acoustic_model",
    "waveform_architecture",
    "waveform_family",
    "representation",
    "training_language_scope",
    "taxonomy_confidence",
    "taxonomy_source",
]
OPTIONAL_TAXONOMY_FIELDS = ["taxonomy_notes"]
EMPTY_MARKERS = {"", "nan", "none", "null", "unknown", "n/a", "na", "?"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def ensure_packages() -> None:
    required = {"pandas": "pandas", "numpy": "numpy", "pyarrow": "pyarrow"}
    missing = [pip_name for module, pip_name in required.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_packages()
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def mount_drive() -> None:
    try:
        from google.colab import drive  
    except Exception:
        print("[DRIVE] Hors Colab : montage skipped.")
        return
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists():
        print("[DRIVE] Google Drive is already mounted.")
    else:
        drive.mount("/content/drive", force_remount=False)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return " ".join(str(value).strip().split())


def normalized_lower(value: object) -> str:
    return normalize_text(value).lower()


def nonempty(value: object) -> bool:
    return normalized_lower(value) not in EMPTY_MARKERS


def unique_join(values: Iterable[object], limit: int = 100) -> str:
    cleaned = sorted({normalize_text(v) for v in values if nonempty(v)})
    if len(cleaned) > limit:
        return "|".join(cleaned[:limit]) + f"|...(+{len(cleaned)-limit})"
    return "|".join(cleaned)


def reason_tokens(value: object) -> Tuple[str, ...]:
    text = normalize_text(value)
    if not text:
        return tuple()
    return tuple(sorted({token.strip() for token in text.split(";") if token.strip()}))


def is_duration_only_failure(value: object) -> bool:
    return reason_tokens(value) == ("duration_ratio_out_of_bounds",)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def read_parquet_file(path: Path) -> pd.DataFrame:

    return pq.read_table(path).to_pandas()


def atomic_parquet_dump(dataframe: pd.DataFrame, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    pq.write_table(table, tmp, compression="snappy")
    os.replace(tmp, path)


def locate_phase0_manifest() -> Tuple[Path, str]:
    direct = PHASE0_DIR / "mlaad_mailabs_manifest_v2_2_canonical.parquet"
    if direct.is_file():
        return direct.resolve(), "phase0_directory"

    root = Path("/content/drive/MyDrive/fingerprint_q1_outputs/phase0")
    if root.exists():
        matches = sorted(root.rglob("mlaad_mailabs_manifest_v2_2_canonical.parquet"))
        if matches:
            return matches[0].resolve(), "phase0_drive_search"

    archives: List[Path] = []
    if str(PHASE0_ZIP).strip() and PHASE0_ZIP.is_file():
        archives.append(PHASE0_ZIP)
    mydrive = Path("/content/drive/MyDrive")
    if mydrive.exists():
        archives.extend(sorted(mydrive.rglob("phase0_mlaad_mailabs_v2_3_2*.zip")))

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    for archive_path in archives:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                members = [name for name in archive.namelist()
                           if name.endswith("mlaad_mailabs_manifest_v2_2_canonical.parquet")]
                if not members:
                    continue
                member = sorted(members)[0]
                target = RUNTIME_DIR / "mlaad_mailabs_manifest_v2_2_canonical.parquet"
                with archive.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                return target, f"phase0_zip:{archive_path}"
        except zipfile.BadZipFile:
            continue

    raise FileNotFoundError(
        "Phase 0 manifest not found. Check PHASE0_DIR or place "
        "l'archive phase0_mlaad_mailabs_v2_3_2*.zip in MyDrive."
    )


def validate_phase0(dataframe: pd.DataFrame) -> Dict[str, int]:
    required = {
        "pair_id", "independent_generator_id", "language", "qc_status",
        "exclusion_reason", "fake_path", "real_path", "fake_sha256",
        "real_sha256", "model_name_meta", "architecture_meta",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(f"Missing Phase 0 columns: {missing}")
    if dataframe["pair_id"].astype(str).duplicated().any():
        raise RuntimeError("The Phase 0 manifest contains duplicate pair_id values.")

    report = {
        "n_pairs": int(len(dataframe)),
        "n_generators": int(dataframe["independent_generator_id"].astype(str).nunique()),
        "n_languages": int(dataframe["language"].astype(str).nunique()),
        "n_qc_ok": int(dataframe["qc_status"].astype(str).str.lower().eq("ok").sum()),
    }
    if report["n_pairs"] != EXPECTED_TOTAL_PAIRS:
        raise RuntimeError(
            f"Incorrect population: {report['n_pairs']:,}; "
            f"{EXPECTED_TOTAL_PAIRS:,} expected."
        )
    if report["n_generators"] != EXPECTED_GENERATORS:
        raise RuntimeError(
            f"{report['n_generators']} generators; {EXPECTED_GENERATORS} expected."
        )
    if report["n_languages"] != EXPECTED_LANGUAGES:
        raise RuntimeError(
            f"{report['n_languages']} languages; {EXPECTED_LANGUAGES} expected."
        )
    return report


def build_inventory(dataframe: pd.DataFrame) -> pd.DataFrame:
    work = dataframe.copy()
    work["strict_ok"] = work["qc_status"].astype(str).str.lower().eq("ok")
    work["duration_only_fail"] = work["exclusion_reason"].map(is_duration_only_failure)
    rows: List[Dict[str, Any]] = []
    for generator, group in work.groupby("independent_generator_id", sort=True):
        rows.append({
            "independent_generator_id": str(generator),
            "n_candidates": int(len(group)),
            "n_qc_ok_strict": int(group["strict_ok"].sum()),
            "n_duration_only_fail": int(group["duration_only_fail"].sum()),
            "n_languages": int(group["language"].astype(str).nunique()),
            "languages": "|".join(sorted(group["language"].astype(str).unique())),
            "model_name_meta_values": unique_join(group["model_name_meta"]),
            "architecture_meta_values": unique_join(group["architecture_meta"]),
        })
    return pd.DataFrame(rows).sort_values("independent_generator_id").reset_index(drop=True)


def build_taxonomy(inventory: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    taxonomy = pd.DataFrame(EMBEDDED_TAXONOMY).copy()
    taxonomy["independent_generator_id"] = taxonomy["independent_generator_id"].map(normalize_text)
    if taxonomy["independent_generator_id"].duplicated().any():
        raise RuntimeError("Duplicate identifiers in the integrated taxonomy.")

    expected = set(inventory["independent_generator_id"].astype(str))
    observed = set(taxonomy["independent_generator_id"].astype(str))
    missing_ids = sorted(expected - observed)
    extra_ids = sorted(observed - expected)
    if missing_ids or extra_ids:
        raise RuntimeError(
            f"Taxonomy incompatible. Missing={missing_ids}; additional={extra_ids}"
        )

    for field in TAXONOMY_FIELDS + OPTIONAL_TAXONOMY_FIELDS:
        if field not in taxonomy.columns:
            taxonomy[field] = ""
        taxonomy[field] = taxonomy[field].map(normalize_text)

    incomplete = taxonomy.loc[
        ~taxonomy[TAXONOMY_FIELDS].apply(
            lambda row: all(nonempty(value) for value in row), axis=1
        )
    ]
    if len(incomplete):
        raise RuntimeError(
            "Taxonomy integrated incomplete for : "
            + ", ".join(incomplete["independent_generator_id"].astype(str))
        )

    invalid_conf = taxonomy.loc[
        ~taxonomy["taxonomy_confidence"].str.lower().isin(VALID_CONFIDENCE),
        "independent_generator_id",
    ].tolist()
    if invalid_conf:
        raise RuntimeError(f"taxonomy_confidence invalid: {invalid_conf}")

    taxonomy["taxonomy_confidence"] = taxonomy["taxonomy_confidence"].str.lower()
    taxonomy["taxonomy_complete"] = True
    taxonomy["confirmatory_confidence"] = taxonomy["taxonomy_confidence"].isin(
        CONFIRMATORY_CONFIDENCE
    )

    full_csv = taxonomy.merge(
        inventory,
        on="independent_generator_id",
        how="left",
        validate="one_to_one",
    )
    ordered_columns = [
        "independent_generator_id", *TAXONOMY_FIELDS,
        *OPTIONAL_TAXONOMY_FIELDS,
        "n_candidates", "n_qc_ok_strict", "n_duration_only_fail",
        "n_languages", "languages", "model_name_meta_values",
        "architecture_meta_values",
    ]
    full_csv = full_csv[ordered_columns]

    TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TAXONOMY_PATH.exists() and CREATE_BACKUP_BEFORE_OVERWRITE:
        backup = TAXONOMY_PATH.with_name(
            TAXONOMY_PATH.stem + ".backup_before_v1_1.csv"
        )
        if not backup.exists():
            shutil.copy2(TAXONOMY_PATH, backup)
            print(f"[TAXONOMY] Saved: {backup}")
    if OVERWRITE_TAXONOMY_CSV or not TAXONOMY_PATH.exists():
        full_csv.to_csv(TAXONOMY_PATH, index=False, encoding="utf-8-sig")

    report = {
        "taxonomy_path": str(TAXONOMY_PATH),
        "n_expected_generators": len(expected),
        "n_complete_generators": int(taxonomy["taxonomy_complete"].sum()),
        "n_confirmatory_confidence": int(taxonomy["confirmatory_confidence"].sum()),
        "confidence_counts": {str(k): int(v) for k, v in taxonomy["taxonomy_confidence"].value_counts().items()},
        "family_counts": {str(k): int(v) for k, v in taxonomy["waveform_family"].value_counts().items()},
        "all_55_complete": bool(len(taxonomy) == EXPECTED_GENERATORS),
    }
    return taxonomy, report


def merge_taxonomy(dataframe: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    remove = [
        *TAXONOMY_FIELDS, *OPTIONAL_TAXONOMY_FIELDS,
        "taxonomy_status", "taxonomy_complete", "confirmatory_confidence",
    ]
    work = dataframe.drop(columns=[c for c in remove if c in dataframe.columns]).copy()
    merged = work.merge(
        taxonomy[[
            "independent_generator_id", *TAXONOMY_FIELDS,
            *OPTIONAL_TAXONOMY_FIELDS, "taxonomy_complete",
            "confirmatory_confidence",
        ]],
        on="independent_generator_id",
        how="left",
        validate="many_to_one",
    )
    if not merged["taxonomy_complete"].fillna(False).all():
        bad = sorted(merged.loc[
            ~merged["taxonomy_complete"].fillna(False),
            "independent_generator_id",
        ].astype(str).unique())
        raise RuntimeError(f"Taxonomy missing after merge : {bad}")
    merged["taxonomy_status"] = "complete"
    return merged


def protocol_masks(dataframe: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    strict = dataframe["qc_status"].astype(str).str.lower().eq("ok")
    duration_only = dataframe["exclusion_reason"].map(is_duration_only_failure)
    paths_present = dataframe["fake_path"].map(nonempty) & dataframe["real_path"].map(nonempty)
    hashes_present = dataframe["fake_sha256"].map(nonempty) & dataframe["real_sha256"].map(nonempty)
    relaxed = strict | (duration_only & paths_present & hashes_present)
    return strict, relaxed


def make_manifest(merged: pd.DataFrame, mask: pd.Series, protocol: str) -> pd.DataFrame:
    output = merged.loc[mask & merged["taxonomy_complete"].fillna(False)].copy()
    output["qc_protocol"] = protocol
    output["was_reincluded_by_relaxed_protocol"] = (
        protocol == "relaxed"
    ) & ~output["qc_status"].astype(str).str.lower().eq("ok")
    return output.reset_index(drop=True)


def eligibility(
    manifest: pd.DataFrame, protocol: str
) -> Tuple[pd.DataFrame, pd.DataFrame, set[str], set[str]]:
    generator_counts = (
        manifest.groupby(
            ["waveform_family", "independent_generator_id", "taxonomy_confidence"],
            dropna=False, observed=True,
        )
        .agg(
            n_pairs=("pair_id", "size"),
            n_languages=("language", "nunique"),
            languages=("language", lambda x: "|".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    generator_counts["protocol"] = protocol
    generator_counts["meets_minimum_pairs"] = generator_counts["n_pairs"].ge(
        MIN_PAIRS_PER_GENERATOR
    )
    generator_counts["confidence_confirmatory"] = generator_counts[
        "taxonomy_confidence"
    ].astype(str).str.lower().isin(CONFIRMATORY_CONFIDENCE)
    generator_counts["generator_confirmatory_eligible"] = (
        generator_counts["meets_minimum_pairs"]
        & generator_counts["confidence_confirmatory"]
    )

    eligible_generators_frame = generator_counts.loc[
        generator_counts["generator_confirmatory_eligible"]
    ].copy()
    family_rows: List[Dict[str, Any]] = []
    for family, group in eligible_generators_frame.groupby(
        "waveform_family", sort=True, dropna=False
    ):
        n_generators = int(group["independent_generator_id"].nunique())
        family_rows.append({
            "protocol": protocol,
            "waveform_family": normalize_text(family),
            "n_generators_eligible": n_generators,
            "n_pairs_eligible": int(group["n_pairs"].sum()),
            "n_languages": int(manifest.loc[
                manifest["waveform_family"].astype(str).eq(str(family)), "language"
            ].nunique()),
            "generators": "|".join(sorted(group["independent_generator_id"].astype(str).unique())),
            "family_confirmatory_eligible": n_generators >= MIN_GENERATORS_PER_FAMILY,
        })

    family_counts = pd.DataFrame(family_rows)
    if family_counts.empty:
        family_counts = pd.DataFrame(columns=[
            "protocol", "waveform_family", "n_generators_eligible",
            "n_pairs_eligible", "n_languages", "generators",
            "family_confirmatory_eligible",
        ])

    families = set(family_counts.loc[
        family_counts["family_confirmatory_eligible"], "waveform_family"
    ].astype(str))
    generators = set(eligible_generators_frame["independent_generator_id"].astype(str))
    return generator_counts, family_counts, families, generators


def create_audit_bundle(output_dir: Path) -> Path:
    bundle = output_dir / "phase0b_mlaad_taxonomy_v1_2_new_story_audit_bundle.zip"
    names = [
        "phase0b_summary.json",
        "generator_inventory.csv",
        "taxonomy_validation.csv",
        "generator_eligibility_by_protocol.csv",
        "family_eligibility_by_protocol.csv",
        "manifest_counts_by_protocol.csv",
    ]
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            path = output_dir / name
            if path.exists():
                archive.write(path, arcname=name)
    return bundle


def main() -> None:
    mount_drive()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("PHASE 0B — INTEGRATED TAXONOMY + PHASE 1 MANIFESTS")
    print("No audio file will be reread or recomputed.")
    print("=" * 100)

    manifest_path, source_mode = locate_phase0_manifest()
    print(f"[SOURCE] {manifest_path}")
    phase0 = read_parquet_file(manifest_path)
    phase0_report = validate_phase0(phase0)
    print(f"[PHASE 0] {phase0_report}")

    inventory = build_inventory(phase0)
    inventory.to_csv(OUTPUT_DIR / "generator_inventory.csv", index=False, encoding="utf-8-sig")

    taxonomy, taxonomy_report = build_taxonomy(inventory)
    print(f"[TAXONOMY] {taxonomy_report}")
    validation = taxonomy[[
        "independent_generator_id", "taxonomy_complete",
        "confirmatory_confidence", "taxonomy_confidence",
        "waveform_family", "waveform_architecture",
    ]].copy()
    validation.to_csv(OUTPUT_DIR / "taxonomy_validation.csv", index=False, encoding="utf-8-sig")

    merged = merge_taxonomy(phase0, taxonomy)
    strict_mask, relaxed_mask = protocol_masks(merged)

    strict_all = make_manifest(merged, strict_mask, "strict")
    relaxed_all = make_manifest(
        merged,
        relaxed_mask if BUILD_RELAXED_PROTOCOL else strict_mask,
        "relaxed" if BUILD_RELAXED_PROTOCOL else "strict",
    )

    strict_gen, strict_family, strict_families, strict_generators = eligibility(
        strict_all, "strict"
    )
    relaxed_gen, relaxed_family, relaxed_families, relaxed_generators = eligibility(
        relaxed_all, "relaxed"
    )

    strict_confirmatory = strict_all.loc[
        strict_all["waveform_family"].astype(str).isin(strict_families)
        & strict_all["independent_generator_id"].astype(str).isin(strict_generators)
    ].copy()
    relaxed_confirmatory = relaxed_all.loc[
        relaxed_all["waveform_family"].astype(str).isin(relaxed_families)
        & relaxed_all["independent_generator_id"].astype(str).isin(relaxed_generators)
    ].copy()

    generator_eligibility = pd.concat([strict_gen, relaxed_gen], ignore_index=True)
    family_eligibility = pd.concat([strict_family, relaxed_family], ignore_index=True)
    generator_eligibility.to_csv(
        OUTPUT_DIR / "generator_eligibility_by_protocol.csv", index=False, encoding="utf-8-sig"
    )
    family_eligibility.to_csv(
        OUTPUT_DIR / "family_eligibility_by_protocol.csv", index=False, encoding="utf-8-sig"
    )

    outputs = {
        "mlaad_phase1_strict_all_taxonomized.parquet": strict_all,
        "mlaad_phase1_relaxed_all_taxonomized.parquet": relaxed_all,
        "mlaad_phase1_strict_confirmatory.parquet": strict_confirmatory,
        "mlaad_phase1_relaxed_confirmatory.parquet": relaxed_confirmatory,
    }
    for name, frame in outputs.items():
        atomic_parquet_dump(frame, OUTPUT_DIR / name)

    counts = pd.DataFrame([
        {"protocol": "strict_all_taxonomized", "n_pairs": len(strict_all),
          "n_generators": strict_all["independent_generator_id"].nunique(),
          "n_families": strict_all["waveform_family"].nunique(),
          "n_languages": strict_all["language"].nunique()},
        {"protocol": "relaxed_all_taxonomized", "n_pairs": len(relaxed_all),
          "n_generators": relaxed_all["independent_generator_id"].nunique(),
          "n_families": relaxed_all["waveform_family"].nunique(),
          "n_languages": relaxed_all["language"].nunique()},
        {"protocol": "strict_confirmatory", "n_pairs": len(strict_confirmatory),
          "n_generators": strict_confirmatory["independent_generator_id"].nunique(),
          "n_families": strict_confirmatory["waveform_family"].nunique(),
          "n_languages": strict_confirmatory["language"].nunique()},
        {"protocol": "relaxed_confirmatory", "n_pairs": len(relaxed_confirmatory),
          "n_generators": relaxed_confirmatory["independent_generator_id"].nunique(),
          "n_families": relaxed_confirmatory["waveform_family"].nunique(),
          "n_languages": relaxed_confirmatory["language"].nunique()},
    ])
    counts.to_csv(OUTPUT_DIR / "manifest_counts_by_protocol.csv", index=False, encoding="utf-8-sig")

    summary = {
        "version": VERSION,
        "status": "COMPLETE",
        "source_mode": source_mode,
        "phase0_manifest": str(manifest_path),
        "phase0_manifest_sha256": sha256_file(manifest_path),
        "phase0": phase0_report,
        "taxonomy": taxonomy_report,
        "parameters": {
            "min_generators_per_family": MIN_GENERATORS_PER_FAMILY,
            "min_pairs_per_generator": MIN_PAIRS_PER_GENERATOR,
            "confirmatory_confidence": sorted(CONFIRMATORY_CONFIDENCE),
            "build_relaxed_protocol": BUILD_RELAXED_PROTOCOL,
        },
        "strict": {
            "n_pairs_all_taxonomized": int(len(strict_all)),
            "n_pairs_confirmatory": int(len(strict_confirmatory)),
            "eligible_families": sorted(strict_families),
            "n_reincluded_duration_only": 0,
        },
        "relaxed": {
            "n_pairs_all_taxonomized": int(len(relaxed_all)),
            "n_pairs_confirmatory": int(len(relaxed_confirmatory)),
            "eligible_families": sorted(relaxed_families),
            "n_reincluded_duration_only": int(
                relaxed_all["was_reincluded_by_relaxed_protocol"].sum()
            ),
        },
        "output_directory": str(OUTPUT_DIR),
    }
    atomic_json_dump(summary, OUTPUT_DIR / "phase0b_summary.json")
    bundle = create_audit_bundle(OUTPUT_DIR)
    atomic_json_dump({
        "version": VERSION,
        "status": "COMPLETE",
        "summary": str(OUTPUT_DIR / "phase0b_summary.json"),
        "audit_bundle": str(bundle),
    }, OUTPUT_DIR / ".PHASE0B_COMPLETE.json")

    print("\n" + "=" * 100)
    print("PHASE 0B COMPLETE")
    print(f"Taxonomy complete                  : {len(taxonomy)} / {EXPECTED_GENERATORS}")
    print(f"STRICT — all taxonomized       : {len(strict_all):,}")
    print(f"STRICT — confirmatory             : {len(strict_confirmatory):,}")
    print(f"RELAXED — all taxonomized      : {len(relaxed_all):,}")
    print(f"RELAXED — confirmatory            : {len(relaxed_confirmatory):,}")
    print(
        "Reincluded pairs (ratio only)    :",
        int(relaxed_all["was_reincluded_by_relaxed_protocol"].sum()),
    )
    print("Families confirmatory STRICT     :", sorted(strict_families))
    print("Families confirmatory RELAXED    :", sorted(relaxed_families))
    print("Taxonomy CSV                      :", TAXONOMY_PATH)
    print("Outputs                            :", OUTPUT_DIR)
    print("Archive d'audit                    :", bundle)
    print("=" * 100)


if __name__ == "__main__":
    main()
