# Sources and attribution

## Research fixture and contract

Source: https://github.com/vIlkAm/tribe-research
Commit: 65ab40704ca484b453e2e2675d777423f325f97a.
The fixture is synthetic dry-run data; source clips and raw model outputs are
not included. The project is a non-commercial research demonstration.

## TRIBE v2

Visual reference and model context: https://aidemos.atmeta.com/tribev2
Code: https://github.com/facebookresearch/tribev2
d'Ascoli et al., A Foundation Model of Vision, Audition, and Language for
In-Silico Neuroscience (2026). TRIBE v2 is CC-BY-NC-4.0. No TRIBE model weights
or Meta website source code are redistributed by this frontend.

## Brain anatomy

FreeSurfer fsaverage5 pial surfaces and sulcal depth, distributed by Nilearn 0.12.1:
https://github.com/nilearn/nilearn/tree/0.12.1/nilearn/datasets/data/fsaverage5

FreeSurfer: https://surfer.nmr.mgh.harvard.edu/
Fischl, B., Sereno, M. I., and Dale, A. M. (1999), Cortical surface-based analysis.
II: Inflation, flattening, and a surface-based coordinate system. NeuroImage.

## Region annotations

HCP-MMP1, Glasser et al. (2016), A multi-modal parcellation of human cerebral
cortex. Nature 536:171–178. https://doi.org/10.1038/nature18933

fsaverage annotation: https://doi.org/10.6084/m9.figshare.3498446
Download mirror: https://github.com/tannerjared/HCP-MMP1
Hashes verified against the research repository's ROI-map builder.

## Frontend libraries

React, Three.js, Vite, and Lucide are installed via npm. Their licenses are
included with the installed packages. DM Sans and IBM Plex Mono use the SIL
Open Font License and are served by Google Fonts with local system fallbacks.

## User-selected YouTube references

The interactive comparison embeds the original YouTube players for:
- Rolltreppe KONE TravelMaster 110 im Westfield Mega Mall Hamburg:
  https://www.youtube.com/watch?v=J5Mv-BehKY4 (8–16s).
- The Big Bang Theory - Intro (Deutsch) 1080p HD:
  https://www.youtube.com/watch?v=8FKGDKEwGf0 (from 5s through the end).

The videos and audio remain with their respective owners and are not included in
this package. These references are not presented as ViralBrain-created edits.
Playback uses the official YouTube IFrame Player API:
https://developers.google.com/youtube/iframe_api_reference
