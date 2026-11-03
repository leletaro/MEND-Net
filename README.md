# MEND-Net: Gated Long-Range Feature Learning for Airport Pavement Damage Detection under Variable Illumination

**Official implementation and reproducibility materials for MEND-Net**, a YOLOv8n-based framework for airport pavement damage detection under variable illumination, including challenging nighttime inspection conditions.

> **Release preparation note:** This repository is intended to publish the exact implementation used for the reported experiments. Before public release, confirm the final model configuration, source-code integration, training entry point, and evaluation protocol against the trained checkpoints. The documentation below describes the intended release; it must not be interpreted as evidence that a reproduction has already been verified.

## 1. Overview

Airport pavement damage detection is challenging when damage boundaries are elongated, fragmented, or weakly visible, especially under nonuniform nighttime illumination. MEND-Net integrates three components into a single-stage YOLOv8n detection framework:

| Network stage | Component | Function in MEND-Net |
| --- | --- | --- |
| **Feature Extraction Network** | **MiGLA-CPE** | Integrates long-range feature aggregation, local positional information, and feature modulation in the high-level representation. |
| **Multi-Scale Feature Fusion Network** | **LSKA** | Uses separable large-kernel spatial filtering to enrich contextual information. |
| **Objective Detection Network** | **DyHead** | Refines multi-scale detection representations across the P3–P5 feature levels. |

The methodological contribution is the task-specific design and integration of these components, **not** the invention of the original LSKA, DyHead, or MLLA architectures. Their sources are acknowledged below and in [`docs/MODULE_PROVENANCE.md`](docs/MODULE_PROVENANCE.md).

## 2. Architecture and implementation disclosure

The planned publication configuration includes:

- **Base detector:** YOLOv8n, using a modified Ultralytics codebase.
- **MiGLA-CPE:** high-level feature extraction block (P5). The published implementation must identify its query/key feature mapping, any attention normalization, positional convolution, feature gate, and residual connection exactly as implemented.
- **LSKA:** P5 contextual module with a nominal **11 × 11** effective kernel and a residual configuration. The code and documentation must specify the actual sequence of separable convolutions, padding, dilation, and any learnable scaling.
- **DyHead:** three detection refinement layers consuming the **P3, P4, and P5** features. The actual adjacent-scale fusion coefficient must be recorded per released experiment in the model configuration or training script.

**Important configuration distinction.** The comparison experiment named `APDD_DyHead` uses **α = 0.50** even if that value is omitted from its run-directory name. The historical E10-Full configuration also used **α = 0.50**. The selected training script originally named `Trainv11UP_MEND_a075_300.py` suggests a potentially different configuration (**α = 0.75**) by filename, but filenames do not establish runtime settings. **Do not report a single MEND-Net α value until the selected script and checkpoint have been checked.** Document the released MEND-Net value here after verification: **`[VERIFY FROM CODE AND CHECKPOINT]`**.

For the implemented equations, upstream correspondence, and modification-level comparison, see [`docs/MODULE_PROVENANCE.md`](docs/MODULE_PROVENANCE.md). Do not replace these with conceptual equations that do not exactly describe the implementation.

## 3. APDD-DNV dataset

APDD-DNV (*Airport Pavement Damage Dataset with Diurnal-Nocturnal Variants*) includes six airport concrete-pavement damage categories:

`Joint_deterioration`, `Corner_break`, `Crack`, `Grouting_patching`, `Corner_pothole`, and `Localized_patching`.

The dataset configurations and splits used in the revised experimental protocol are:

| Configuration | Train | Validation | Independent test |
| --- | ---: | ---: | ---: |
| APDD-Day | 672 | 155 | 155 |
| APDD-Night | 846 | 192 | 192 |
| APDD-MultiIllum (V2) | 935 | 80 | 215 |

**APDD-MultiIllum V2 uses real daytime and real nighttime images; it does not include synthetically darkened nighttime images.** See the manuscript for the source-image allocation and airport-specific partitions. The same source image must not be assigned to multiple splits *within an evaluated configuration*. Images reused across different configurations must be disclosed rather than counted as additional independent originals.

Data files are not included here unless their redistribution is explicitly authorized. A reproducible release should provide, subject to permissions, the exact split manifests, class definitions, preprocessing instructions, and dataset-YAML examples. **Do not commit private image data or absolute Windows paths.**

## 4. Proposed repository organization

```text
MEND-Net/
├── README.md
├── LICENSE                         # AGPL-3.0; preserve upstream notices
├── .gitignore
├── train_mendnet.py                # the ONE retained training entry point
├── APDD-DNV_day.yaml              # portable dataset configuration
├── APDD-DNV_night.yaml            # portable dataset configuration
├── APDD-DNV_mix.yaml              # portable dataset configuration
├── yaml-series/                   # only the model YAML files actually used
├── ultralytics/                   # corresponding modified framework source
├── docs/
│   ├── MODULE_PROVENANCE.md       # upstream sources and exact modifications
│   ├── RELEASE_CHECKLIST.md
│   └── RESPONSE_R3_7.md
└── [evaluation script and split manifests, when verified]
```

The directory tree is a **proposed release layout**, not a verified inventory of the local folder. In particular, if `Model/` or `initializations/` contains required imported source modules, initial weights, or architecture definitions, keep those required files and update the directory tree accordingly. Conversely, omit exploratory training scripts, temporary reports, local caches, and redundant copies.

## 5. Environment and training

The recorded experimental setup includes **Ultralytics 8.4.3**, **image size 896**, **batch size 16**, **300 epochs**, **random seed 42**, and **initial learning rate 0.003**. A compatible PyTorch/CUDA/MMCV environment must be documented from the machine that produced the released checkpoints; do not assume these dependencies can be reconstructed from the Ultralytics version alone.

Before launching, replace private absolute paths inside the renamed script and dataset YAML files with project-relative paths or documented user-configurable paths.

From the repository root, check that Python imports the **local modified** framework:

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

If `train_mendnet.py` preserves a direct script entry point, the intended training command is:

```bash
python train_mendnet.py
```

This command is **not yet verified** against the unpublished local code. Run a smoke test and save the final command, CLI arguments (if any), dependency versions, model YAML, data YAML, checkpoint metadata, and seed in the release documentation.

**Experimental reporting:** validation results are not interchangeable with independent-test results. Report the dataset configuration, split, number of images, checkpoint, and evaluation settings alongside every published metric.

## 6. Model evaluation and checkpoint provenance

For scientific reproducibility, the public release should provide:

1. A documented evaluation entry point and the exact validation/test options (including `imgsz`, `batch`, `rect`, confidence, and IoU settings).
2. The independent test-set manifest and source-image provenance, when redistribution is permitted.
3. The final selected checkpoint or an accessible distribution location, with its SHA-256 checksum, if redistribution is permitted.
4. The precise architecture YAML, DyHead fusion setting, MiGLA-CPE gate/kernel implementation, and the checkpoint's training configuration.
5. A table distinguishing the baseline, each ablation, the matched DyHead-only comparison, and the final MEND-Net model.

If weights or images cannot be distributed, clearly state that restriction and specify which numerical results cannot be independently reproduced from this repository alone.

## 7. Upstream implementations and attribution

MEND-Net integrates and modifies ideas originating from the following work:

- **Ultralytics YOLOv8** — [official repository](https://github.com/ultralytics/ultralytics). Distributed under AGPL-3.0 for the open-source variant.
- **Dynamic Head (DyHead)** — X. Dai *et al.*, *Dynamic Head: Unifying Object Detection Heads with Attentions*, CVPR 2021. [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_Dynamic_Head_Unifying_Object_Detection_Heads_With_Attentions_CVPR_2021_paper.html) · [Official implementation](https://github.com/microsoft/DynamicHead) (MIT).
- **Large Separable Kernel Attention (LSKA)** — K. W. Lau, L.-M. Po, and Y. A. U. Rehman, *Large Separable Kernel Attention: Rethinking the Large Kernel Attention Design in CNN*, *Expert Systems with Applications*. [Paper](https://doi.org/10.1016/j.eswa.2023.121352) · [Author implementation](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention) (Apache-2.0).
- **Mamba-Like Linear Attention (MLLA; later called MILA)** — D. Han *et al.*, *Demystify Mamba in Vision: A Linear Attention Perspective*, NeurIPS 2024. [Paper](https://papers.nips.cc/paper_files/paper/2024/hash/e618724ac897c6cf3fbfb273f8695d67-Abstract-Conference.html) · [Official implementation](https://github.com/LeapLabTHU/MLLA) (MIT).

**Attribution does not by itself prove code ancestry.** For each adapted module, cite the actual source file or repository commit used, retain applicable notices, and state the exact changes in `docs/MODULE_PROVENANCE.md`. If a local `reference/` file originated from another repository, verify its provenance and licensing before redistributing it.

## 8. License and citation

This project is derived from the open-source Ultralytics implementation. For public release, include the complete **GNU Affero General Public License v3.0 (AGPL-3.0)** and preserve applicable third-party license and attribution notices. Verify that the published code and included artifacts conform to all upstream license obligations.

**Paper citation:** add the final bibliographic reference after manuscript acceptance/publication. In the meantime, cite the associated manuscript by its actual public preprint or DOI **only if one exists**; do not invent a publication identifier.

## 9. Release status

This README is a publication-ready **draft template**, not a claim that the source files, model checkpoints, or independent test data have already been released. Complete and verify the items in [`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md) before linking the repository in a reviewer response.
