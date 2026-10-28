# RAVI results: colab_survey

## Environment of the benchmark stage

- **time**: 2026-10-26 09:57:47
- **cpu_model**: Intel(R) Xeon(R) CPU @ 2.20GHz
- **cpu_count**: 12
- **ram_gb**: 83.5
- **gpu**: NVIDIA A100-SXM4-40GB
- **faiss**: 1.15.1
- **faiss_max_threads**: 6
- **torch**: 2.11.0+cu130
- **python**: 3.13.16
- **platform**: Linux-6.6.122+-x86_64-with-glibc2.39

Timing numbers (QPS, latency) are only comparable within this environment. Distance evaluations per query (`ndis`) do not depend on the hardware.

## Setup

- dataset: food101
- backbones: resnet50.a1_in1k, convnext_base.fb_in22k_ft_in1k, vit_base_patch16_224.augreg_in21k_ft_in1k, vit_base_patch16_clip_224.openai, vit_base_patch14_dinov2.lvd142m, vit_base_patch16_224.mae
- reduced dimensions: [64, 128, 256]; reductions: ['pca', 'head_rank', 'head_hub', 'head_koleo']
- benchmark seeds: [0, 1]; recall targets: [0.8, 0.9, 0.95]

## 1. Neighborhoods kept after reduction (exact search)

Mean over backbones of recall@10 of exact search in the reduced space against exact search in the native space. 1.0 means the reduction keeps every true neighbor.

| reduction | d=64 | d=128 | d=256 |
|---|---|---|---|
| head_hub |  |  | 0.509 |
| head_koleo |  |  | 0.577 |
| head_rank |  |  | 0.572 |
| pca | 0.541 | 0.631 | 0.683 |


Class precision@10 of exact search (share of the 10 results with the query's label), per backbone:

| model | native | head_hub256 | head_koleo256 | head_rank256 | pca128 | pca256 | pca64 |
|---|---|---|---|---|---|---|---|
| convnext_base.fb_in22k_ft_in1k | 0.775 | 0.761 | 0.770 | 0.769 | 0.764 | 0.771 | 0.745 |
| resnet50.a1_in1k | 0.419 | 0.410 | 0.415 | 0.414 | 0.406 | 0.420 | 0.389 |
| vit_base_patch14_dinov2.lvd142m | 0.823 | 0.810 | 0.822 | 0.818 | 0.818 | 0.821 | 0.808 |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 0.772 | 0.748 | 0.756 | 0.755 | 0.765 | 0.771 | 0.738 |
| vit_base_patch16_224.mae | 0.051 | 0.112 | 0.164 | 0.143 | 0.062 | 0.062 | 0.057 |
| vit_base_patch16_clip_224.openai | 0.793 | 0.732 | 0.745 | 0.748 | 0.797 | 0.800 | 0.782 |


## 2a. Native spaces: cost at index recall@10 >= 0.9

Best build setting of each index type; mean over seeds. `ndis` = distance evaluations per query (lower is better, hardware-independent); `qps_1t` = single-thread queries per second.

| model | dim | index | build_key | ndis | ndis_std | qps_1t | qps_1t_std | lat_p99_ms | bytes_per_vec | build_time_s |
|---|---|---|---|---|---|---|---|---|---|---|
| convnext_base.fb_in22k_ft_in1k | 1024 | flat | flat | 75750.000 |  | 38.216 |  | 26.940 | 4096.001 | 0.209 |
| resnet50.a1_in1k | 2048 | flat | flat | 75750.000 |  | 19.480 |  | 53.387 | 8192.001 | 0.415 |
| vit_base_patch14_dinov2.lvd142m | 768 | flat | flat | 75750.000 |  | 50.828 |  | 20.444 | 3072.001 | 0.158 |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 768 | flat | flat | 75750.000 |  | 50.936 |  | 20.613 | 3072.001 | 0.157 |
| vit_base_patch16_224.mae | 768 | flat | flat | 75750.000 |  | 50.657 |  | 20.381 | 3072.001 | 0.158 |
| vit_base_patch16_clip_224.openai | 512 | flat | flat | 75750.000 |  | 78.176 |  | 13.473 | 2048.001 | 0.104 |
| convnext_base.fb_in22k_ft_in1k | 1024 | hnsw | hnsw_M16_efc200 | 348.595 | 2.610 | 6787.286 | 37.500 | 0.233 | 4240.315 | 10.935 |
| resnet50.a1_in1k | 2048 | hnsw | hnsw_M16_efc200 | 675.474 | 0.104 | 1920.630 | 1.140 | 0.735 | 8336.315 | 27.598 |
| vit_base_patch14_dinov2.lvd142m | 768 | hnsw | hnsw_M16_efc200 | 251.405 | 0.953 | 11685.672 | 189.192 | 0.134 | 3216.315 | 7.931 |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 768 | hnsw | hnsw_M16_efc200 | 379.934 | 1.085 | 7524.650 | 67.915 | 0.213 | 3216.315 | 10.085 |
| vit_base_patch16_224.mae | 768 | hnsw | hnsw_M16_efc200 | 345.424 | 1.681 | 8136.487 | 4.099 | 0.166 | 3216.315 | 8.262 |
| vit_base_patch16_clip_224.openai | 512 | hnsw | hnsw_M16_efc200 | 378.003 | 0.274 | 9687.967 | 188.846 | 0.156 | 2192.315 | 7.922 |
| convnext_base.fb_in22k_ft_in1k | 1024 | ivf | ivf_nlist256 | 1473.148 | 6.877 | 2006.002 | 34.816 | 0.721 | 4117.871 | 2.227 |
| resnet50.a1_in1k | 2048 | ivf | ivf_nlist256 | 5384.139 | 12.001 | 271.230 | 1.361 | 4.706 | 8227.714 | 4.259 |
| vit_base_patch14_dinov2.lvd142m | 768 | ivf | ivf_nlist256 | 1210.512 | 415.353 | 3575.204 | 1345.611 | 0.508 | 3090.411 | 1.724 |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 768 | ivf | ivf_nlist256 | 1501.466 | 10.863 | 2658.434 | 66.867 | 0.523 | 3090.411 | 1.708 |
| vit_base_patch16_224.mae | 768 | ivf | ivf_nlist256 | 1612.266 | 12.029 | 2406.684 | 7.257 | 0.569 | 3090.411 | 1.729 |
| vit_base_patch16_clip_224.openai | 512 | ivf | ivf_nlist256 | 1508.218 | 3.690 | 3942.315 | 4.062 | 0.369 | 2062.950 | 1.237 |


## 2b. All spaces: cost at end-to-end recall@10 >= 0.9 (against the native space)

Median over backbones of the best `ndis` per index; in brackets, how many backbones reached the target at all. A reduced space that cannot reach the target end to end is reported as 0/N.

| space | flat | hnsw | ivf |
|---|---|---|---|
| native | 75,750 [6/6] | 363 [6/6] | 1,505 [6/6] |
| head_hub256 | - [0/6] | - [0/6] | - [0/6] |
| head_koleo256 | - [0/6] | - [0/6] | - [0/6] |
| head_rank256 | - [0/6] | - [0/6] | - [0/6] |
| pca128 | - [0/6] | - [0/6] | - [0/6] |
| pca256 | - [0/6] | - [0/6] | - [0/6] |
| pca64 | - [0/6] | - [0/6] | - [0/6] |


The full per-backbone, per-seed table is results/operating_points.csv.


## 3. Geometry features most associated with log QPS (Spearman, index recall target)

| index | feature | n_spaces | n_models | spearman | ci_lo | ci_hi | partial_spearman_given_dim | q_fdr |
|---|---|---|---|---|---|---|---|---|
| hnsw | qlid_median | 42 | 6 | -0.918 | -0.961 | -0.845 | -0.923 | 0.000 |
| hnsw | lid_mean | 42 | 6 | -0.914 | -0.957 | -0.840 | -0.911 | 0.000 |
| hnsw | lid_median | 42 | 6 | -0.913 | -0.955 | -0.840 | -0.918 | 0.000 |
| hnsw | twonn_id | 42 | 6 | -0.908 | -0.945 | -0.856 | -0.880 | 0.000 |
| hnsw | lid_p90 | 42 | 6 | -0.903 | -0.957 | -0.807 | -0.888 | 0.000 |
| ivf | twonn_id | 42 | 6 | -0.850 | -0.893 | -0.790 | -0.825 | 0.000 |
| ivf | qlid_median | 42 | 6 | -0.829 | -0.882 | -0.765 | -0.836 | 0.000 |
| ivf | lid_median | 42 | 6 | -0.820 | -0.877 | -0.734 | -0.825 | 0.000 |
| ivf | lid_mean | 42 | 6 | -0.814 | -0.885 | -0.717 | -0.798 | 0.000 |
| ivf | lid_p90 | 42 | 6 | -0.800 | -0.873 | -0.704 | -0.759 | 0.000 |
| flat | pca_d90 | 42 | 6 | -0.558 | -0.775 | -0.327 | 0.026 | 0.003 |
| flat | aniso_meancos | 42 | 6 | -0.543 | -0.739 | -0.403 | -0.016 | 0.003 |
| flat | pca_erank | 42 | 6 | -0.503 | -0.750 | -0.247 | -0.009 | 0.006 |
| flat | twonn_id | 42 | 6 | -0.494 | -0.712 | -0.321 | -0.014 | 0.006 |
| flat | lid_std | 42 | 6 | -0.471 | -0.740 | -0.265 | 0.007 | 0.009 |

CI: bootstrap over backbones. With few backbones these intervals are wide; treat them as exploratory.


## 3. Geometry features most associated with log(1/ndis) (Spearman, index recall target)

| index | feature | n_spaces | n_models | spearman | ci_lo | ci_hi | partial_spearman_given_dim | q_fdr |
|---|---|---|---|---|---|---|---|---|
| ivf | clus_silhouette | 42 | 6 | 0.938 | 0.702 | 0.969 | 0.938 | 0.000 |
| hnsw | lid100_mean | 42 | 6 | -0.917 | -0.971 | -0.842 | -0.946 | 0.000 |
| hnsw | qlid_median | 42 | 6 | -0.899 | -0.955 | -0.827 | -0.945 | 0.000 |
| hnsw | lid_median | 42 | 6 | -0.897 | -0.956 | -0.834 | -0.941 | 0.000 |
| hnsw | clus_silhouette | 42 | 6 | 0.894 | 0.662 | 0.988 | 0.890 | 0.000 |
| hnsw | lid_mean | 42 | 6 | -0.882 | -0.950 | -0.808 | -0.930 | 0.000 |
| ivf | qshift_nn | 42 | 6 | -0.845 | -0.878 | -0.336 | -0.852 | 0.000 |
| ivf | lid100_mean | 42 | 6 | -0.761 | -0.912 | -0.496 | -0.790 | 0.000 |
| ivf | qlid_median | 42 | 6 | -0.756 | -0.895 | -0.511 | -0.802 | 0.000 |
| ivf | lid_median | 42 | 6 | -0.745 | -0.888 | -0.479 | -0.788 | 0.000 |
| flat | lid_mean | 42 | 6 |  |  |  | -0.094 |  |
| flat | lid_median | 42 | 6 |  |  |  | -0.103 |  |
| flat | lid_std | 42 | 6 |  |  |  | -0.038 |  |
| flat | lid_p90 | 42 | 6 |  |  |  | -0.085 |  |
| flat | lid100_mean | 42 | 6 |  |  |  | -0.092 |  |

CI: bootstrap over backbones. With few backbones these intervals are wide; treat them as exploratory.


## 4. Reduction methods compared on neighborhood kept (exact search)

| method | mean_rank | mean_value | n_blocks | friedman_chi2 | friedman_p | k |
|---|---|---|---|---|---|---|
| pca | 1.167 | 0.683 | 6 | 11.800 | 0.008 | 4 |
| head_rank | 2.333 | 0.572 | 6 | 11.800 | 0.008 | 4 |
| head_koleo | 2.833 | 0.577 | 6 | 11.800 | 0.008 | 4 |
| head_hub | 3.667 | 0.509 | 6 | 11.800 | 0.008 | 4 |


Pairwise Wilcoxon signed-rank tests, Holm-corrected:

| a | b | mean_diff_a_minus_b | wins_a | wins_b | p | p_holm |
|---|---|---|---|---|---|---|
| head_hub | head_koleo | -0.068 | 1 | 5 | 0.219 | 0.469 |
| head_hub | head_rank | -0.063 | 1 | 5 | 0.156 | 0.469 |
| head_hub | pca | -0.174 | 0 | 6 | 0.031 | 0.188 |
| head_koleo | head_rank | 0.006 | 2 | 4 | 1.000 | 1.000 |
| head_koleo | pca | -0.105 | 0 | 6 | 0.031 | 0.188 |
| head_rank | pca | -0.111 | 1 | 5 | 0.062 | 0.250 |


## 5. Configuration recommender (leave one family out)

Score = efficiency of the chosen configuration / best configuration of that backbone (1 = oracle).

| method | mean_score | mean_regret | worst_score | n_models |
|---|---|---|---|---|
| ours | 1.000 | 0.000 | 1.000 | 6 |
| prior_only | 1.000 | 0.000 | 1.000 | 6 |
| global_best | 1.000 | 0.000 | 1.000 | 6 |
| nn_dataset | 1.000 | 0.000 | 1.000 | 6 |
| random | 0.058 | 0.942 | 0.054 | 6 |
