# Audit status

## Structural pairing validation

The independent structural pairing validation for the MLAAD STRICT confirmatory population was completed. The release code keeps the canonical manifest rebuild and validation scripts under `src/00_pairing/`.

## Physical file audit

The historical full physical audit was only partially completed:

- unique MLAAD/M-AILABS paths in scope: **146,140**;
- paths checked for physical existence/header/SHA integrity: **57,999**;
- failures observed in the checked subset: **0**;
- full physical audit of all unique paths: **not completed**.

This distinction is intentional. A successful structural pairing audit must not be described as a complete physical certification of every audio path.

## Release wording

Permitted wording:

> Independent structural pairing validation was complete for the STRICT confirmatory population; physical existence/header/SHA checks covered 57,999/146,140 unique MLAAD/M-AILABS WAV paths with no failures. The remaining full Phase-0 physical audit was not completed.

Do not replace this with a claim of complete physical verification.
