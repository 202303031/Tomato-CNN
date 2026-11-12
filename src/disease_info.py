"""Short, non-diagnostic descriptions shown in the app. Keyed by normalised display name."""
from __future__ import annotations

import re

_INFO = {
    "bacterial spot": "Caused by Xanthomonas bacteria. Small dark, water-soaked spots that may develop yellow halos; "
                      "spreads in warm, wet conditions.",
    "early blight": "Fungal disease (Alternaria solani). Brown lesions with concentric 'target-like' rings, usually "
                    "starting on older, lower leaves.",
    "late blight": "Oomycete disease (Phytophthora infestans). Large irregular grey-green to brown blotches, often with "
                   "pale edges; spreads rapidly in cool, humid weather.",
    "leaf mold": "Fungal disease (Passalora fulva). Pale yellow patches on the upper leaf surface with olive-green to "
                 "brown mould on the underside; favoured by high humidity.",
    "septoria leaf spot": "Fungal disease (Septoria lycopersici). Many small circular spots with dark borders and "
                          "light centres, typically on lower leaves first.",
    "spider mites two spotted spider mite": "Pest damage from Tetranychus urticae. Fine stippling, yellowing or bronzing "
                                            "of leaves, sometimes with fine webbing.",
    "target spot": "Fungal disease (Corynespora cassiicola). Brown spots with concentric rings resembling a target; "
                   "can resemble early blight.",
    "tomato yellow leaf curl virus": "Viral disease transmitted by whiteflies. Upward curling, yellowing leaf margins "
                                     "and stunted growth.",
    "tomato mosaic virus": "Viral disease causing light/dark green mottling (mosaic), leaf distortion and sometimes "
                           "fern-like leaves.",
    "healthy": "No visible disease symptoms detected in the leaf image.",
}

DISCLAIMER = (
    "⚠️ This is an experimental AI prediction produced by a CNN trained on the PlantVillage dataset "
    "(lab-style photos on plain backgrounds). It is NOT a definitive agricultural diagnosis. "
    "Grad-CAM shows regions that influenced the prediction, not exact lesion boundaries. "
    "Consult an agronomist or plant pathologist before acting on this output."
)


def describe(display_name: str) -> str:
    key = re.sub(r"\s+", " ", display_name.lower()).strip()
    return _INFO.get(key, "No description available for this class.")
