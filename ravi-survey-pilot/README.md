# Representation-Aware Vector Search: pilot study code and measurements

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-GITHUB-USERNAME/ravi-survey-pilot/blob/main/RAVI_Survey_Experiments.ipynb)

This repository accompanies the survey *Representation-Aware Vector Search: A Survey of How Deep Embedding Geometry Shapes Approximate Nearest Neighbor Indexing* (submitted to ACM Computing Surveys). It contains:
- the code of the pilot study in Section 11;
- every measurement from the authors' run;
- the script that turns those measurements into the numbers, tables and figures of the paper.

## Contents

| Path | What it is |
|---|---|
| `RAVI_Survey_Experiments.ipynb` | Self-contained Google Colab notebook that runs the whole pilot study (the code is embedded) |
| `ravi/` | Source of the `ravi` package used by the notebook (version 0.3.0) |
| `configs/colab_survey.yaml` | Configuration of the pilot study (data, encoders, reductions, indexes, seeds) |
| `results/` | Measurements of the authors' run on 26 Oct 2026 (CSV), with the configuration (`config_used.yaml`) and environment (`env.json`) |
| `paper/analyse.py` | Computes every number reported in Section 11 from `results/` |
| `paper/numbers.json` | Output of `analyse.py` for the included results, for comparison |
| `paper/make_fig3.py` | Generates Figure 3. Panel (b) is the exact distance distribution of Gaussian data; panel (c) is a simulation with a fixed seed |
| `tests/` | Unit tests of the metrics, indexes and analysis helpers |

## Reproduce the paper's numbers (no GPU, a few minutes)

```bash
pip install numpy pandas scipy
python paper/analyse.py . my_numbers.json
```

`my_numbers.json` should match `paper/numbers.json`. The table and figure numbers below refer to the submitted version:
- Table 7 (geometry cards);
- Table 8 (status of the predictions);
- Figure 5 (P1, P2, P3);
- the statistics quoted in Section 11.

## Re-run the experiments (Google Colab, GPU)

1. Click **Open in Colab** above.
2. Choose *Runtime → Change runtime type → GPU*.
3. Run all cells.

The notebook does the following:
- downloads Food-101 from Hugging Face (`ethz/food101`; no account needed);
- downloads the six public checkpoints through `timm` and `open_clip`;
- embeds the images and builds 42 compact spaces;
- runs HNSW and IVF with two seeds and writes `survey_results.zip`.

Results are saved to Google Drive, so a disconnect loses nothing: run all cells again and finished stages are skipped. The authors' run used an NVIDIA A100 (40 GB) with Faiss 1.15.1 (`results/env.json`).

## Setup in brief

- **Data:** the 75,750 Food-101 training images form the database. The queries are 20 validation images per class (2,020 images). For the cross-modal test (P2) there are also 808 text queries: 101 class names × 8 prompt templates.
- **Encoders:**
  - ResNet-50 (`resnet50.a1_in1k`);
  - ConvNeXt-B (`convnext_base.fb_in22k_ft_in1k`);
  - ViT-B/16 (`vit_base_patch16_224.augreg_in21k_ft_in1k`);
  - CLIP ViT-B/16 (`vit_base_patch16_clip_224.openai`);
  - DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`);
  - MAE ViT-B/16 (`vit_base_patch16_224.mae`).
- **Compact spaces:** PCA to 64, 128 and 256 dimensions. In addition, three linear heads at 256 dimensions: a ranking head, the same head with a hubness penalty, and the same head with a KoLeo spreading term.
- **Indexes:** Faiss HNSW (M = 16, efConstruction = 200, efSearch 10–1,024) and IVF (256 cells, nprobe 1–256), each with seeds 0 and 1.
- **Search work** is the number of distance evaluations per query (`ndis`) needed to reach a recall@10 target. It is interpolated log-linearly between the two settings that bracket the target and averaged over the seeds.

## Notes

- `results/survey_report.md` was written by the notebook from the nearest grid setting. The paper uses the interpolated values from `paper/analyse.py`, so some cost values differ slightly between the two (for example, ResNet-50 HNSW at recall 0.9: 675 on the grid, 539 interpolated).
- Throughput (QPS) values in `results/report.md` were measured on a shared Colab machine and are not used in the paper; the paper reports hardware-independent distance evaluations.
- Food-101 images and model weights are not redistributed here. They are downloaded from their original sources and remain under their own licences.

## Citation

If you use this code or these measurements, please cite the survey (see `CITATION.cff`).

## License

Code: MIT (see `LICENSE`). Measurements in `results/`: CC BY 4.0.
