# Survey measurements: colab_survey

Benchmark machine: cpu_model=Intel(R) Xeon(R) CPU @ 2.20GHz, cpu_count=12, faiss=1.15.1

Dataset: food101; backbones: 6; seeds: [0, 1]; HNSW M=[16], efConstruction=200; IVF nlist=[256].

## Geometry cards (native space of each backbone)

| model | native_dim | qlid_median | rc_q10 | hub_skew | aniso_meancos | qshift_nn | rr10_pca256 | rr10_sq8 | rr10_bin1 | hnsw_ndis |
|---|---|---|---|---|---|---|---|---|---|---|
| resnet50.a1_in1k | 2048 | 31.8 | 1.45 | 2.62 | 0.418 | 0.992 | 0.623 | 0.98 | 0.152 | 675 |
| convnext_base.fb_in22k_ft_in1k | 1024 | 23.5 | 1.61 | 1.92 | 0.199 | 0.981 | 0.801 | 0.992 | 0.628 | 349 |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 768 | 26.1 | 1.48 | 1.96 | 0.37 | 0.985 | 0.735 | 0.985 | 0.54 | 380 |
| vit_base_patch16_clip_224.openai | 512 | 21.7 | 1.62 | 2.06 | 0.698 | 0.981 | 0.739 | 0.975 | 0.482 | 378 |
| vit_base_patch14_dinov2.lvd142m | 768 | 19 | 1.81 | 1.77 | 0.091 | 0.977 | 0.876 | 0.991 | 0.621 | 251 |
| vit_base_patch16_224.mae | 768 | 18.6 | 3.76 | 1.52 | 0.918 | 0.994 | 0.324 | 0.524 | 0.0646 | 345 |

Full card: results/survey_geometry_cards.csv; LaTeX: results/survey_geometry_cards.tex

## P1: query difficulty and search work at 256 dimensions

| space | model | qlid_median | lid_median | rc_q10 | hnsw_ndis | hnsw_ndis_std | ivf_ndis |
|---|---|---|---|---|---|---|---|
| pca256 | convnext_base.fb_in22k_ft_in1k | 19.5 | 19.7 | 1.76 | 336 | 1.11 | 1.45e+03 |
| pca256 | resnet50.a1_in1k | 24.2 | 24.3 | 1.56 | 647 | 1.06 | 5e+03 |
| pca256 | vit_base_patch14_dinov2.lvd142m | 16.5 | 17 | 1.97 | 246 | 1.39 | 932 |
| pca256 | vit_base_patch16_224.augreg_in21k_ft_in1k | 22 | 22.4 | 1.61 | 365 | 0.801 | 1.45e+03 |
| pca256 | vit_base_patch16_224.mae | 20.9 | 20.8 | 3.24 | 467 | 3.49 | 3.55e+03 |
| pca256 | vit_base_patch16_clip_224.openai | 20.5 | 21.1 | 1.65 | 372 | 0.876 | 1.48e+03 |
| native | convnext_base.fb_in22k_ft_in1k | 23.5 | 23.8 | 1.61 | 349 | 2.61 | 1.47e+03 |
| native | resnet50.a1_in1k | 31.8 | 31.8 | 1.45 | 675 | 0.104 | 5.38e+03 |
| native | vit_base_patch14_dinov2.lvd142m | 19 | 19.4 | 1.81 | 251 | 0.953 | 1.21e+03 |
| native | vit_base_patch16_224.augreg_in21k_ft_in1k | 26.1 | 27.3 | 1.48 | 380 | 1.08 | 1.5e+03 |
| native | vit_base_patch16_224.mae | 18.6 | 18.4 | 3.76 | 345 | 1.68 | 1.61e+03 |
| native | vit_base_patch16_clip_224.openai | 21.7 | 22.4 | 1.62 | 378 | 0.274 | 1.51e+03 |


Spearman correlation across backbones, two-sided permutation p-value. With 6 backbones only |rho| >= 0.89 reaches p < 0.05; treat the result as an illustration, not a test of generality.

| space | predictor | cost | n_backbones | spearman | p_perm |
|---|---|---|---|---|---|
| pca256 | qlid_median | hnsw_ndis | 6 | 0.829 | 0.0583 |
| pca256 | qlid_median | ivf_ndis | 6 | 0.829 | 0.0583 |
| pca256 | lid_median | hnsw_ndis | 6 | 0.771 | 0.103 |
| pca256 | lid_median | ivf_ndis | 6 | 0.771 | 0.103 |
| pca256 | inv_rc_q10 | hnsw_ndis | 6 | 0.371 | 0.497 |
| pca256 | inv_rc_q10 | ivf_ndis | 6 | 0.371 | 0.497 |
| native | qlid_median | hnsw_ndis | 6 | 0.886 | 0.0333 |
| native | qlid_median | ivf_ndis | 6 | 0.257 | 0.658 |
| native | lid_median | hnsw_ndis | 6 | 0.886 | 0.0333 |
| native | lid_median | ivf_ndis | 6 | 0.257 | 0.658 |
| native | inv_rc_q10 | hnsw_ndis | 6 | 0.886 | 0.0333 |
| native | inv_rc_q10 | ivf_ndis | 6 | 0.257 | 0.658 |


## P3: search work at a fixed end-to-end recall as the dimension falls (HNSW)

| model | target | space | dim | ceiling_recall10 | ndis | work |
|---|---|---|---|---|---|---|
| convnext_base.fb_in22k_ft_in1k | 0.8 | pca64 | 64 | 0.579 |  |  |
| convnext_base.fb_in22k_ft_in1k | 0.8 | pca128 | 128 | 0.718 |  |  |
| convnext_base.fb_in22k_ft_in1k | 0.8 | pca256 | 256 | 0.801 | 1.12e+03 | 2.86e+05 |
| convnext_base.fb_in22k_ft_in1k | 0.8 | native | 1024 | 1 | 281 | 2.88e+05 |
| resnet50.a1_in1k | 0.8 | pca64 | 64 | 0.498 |  |  |
| resnet50.a1_in1k | 0.8 | pca128 | 128 | 0.574 |  |  |
| resnet50.a1_in1k | 0.8 | pca256 | 256 | 0.623 |  |  |
| resnet50.a1_in1k | 0.8 | native | 2048 | 1 | 438 | 8.98e+05 |
| vit_base_patch14_dinov2.lvd142m | 0.8 | pca64 | 64 | 0.646 |  |  |
| vit_base_patch14_dinov2.lvd142m | 0.8 | pca128 | 128 | 0.786 |  |  |
| vit_base_patch14_dinov2.lvd142m | 0.8 | pca256 | 256 | 0.876 | 246 | 6.29e+04 |
| vit_base_patch14_dinov2.lvd142m | 0.8 | native | 768 | 1 | 251 | 1.93e+05 |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 0.8 | pca64 | 64 | 0.527 |  |  |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 0.8 | pca128 | 128 | 0.656 |  |  |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 0.8 | pca256 | 256 | 0.735 |  |  |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 0.8 | native | 768 | 1 | 304 | 2.34e+05 |
| vit_base_patch16_224.mae | 0.8 | pca64 | 64 | 0.371 |  |  |
| vit_base_patch16_224.mae | 0.8 | pca128 | 128 | 0.343 |  |  |
| vit_base_patch16_224.mae | 0.8 | pca256 | 256 | 0.324 |  |  |
| vit_base_patch16_224.mae | 0.8 | native | 768 | 1 | 273 | 2.1e+05 |
| vit_base_patch16_clip_224.openai | 0.8 | pca64 | 64 | 0.626 |  |  |
| vit_base_patch16_clip_224.openai | 0.8 | pca128 | 128 | 0.712 |  |  |
| vit_base_patch16_clip_224.openai | 0.8 | pca256 | 256 | 0.739 |  |  |
| vit_base_patch16_clip_224.openai | 0.8 | native | 512 | 1 | 302 | 1.55e+05 |
| convnext_base.fb_in22k_ft_in1k | 0.9 | pca64 | 64 | 0.579 |  |  |
| convnext_base.fb_in22k_ft_in1k | 0.9 | pca128 | 128 | 0.718 |  |  |
| convnext_base.fb_in22k_ft_in1k | 0.9 | pca256 | 256 | 0.801 |  |  |
| convnext_base.fb_in22k_ft_in1k | 0.9 | native | 1024 | 1 | 349 | 3.57e+05 |
| resnet50.a1_in1k | 0.9 | pca64 | 64 | 0.498 |  |  |
| resnet50.a1_in1k | 0.9 | pca128 | 128 | 0.574 |  |  |
| resnet50.a1_in1k | 0.9 | pca256 | 256 | 0.623 |  |  |
| resnet50.a1_in1k | 0.9 | native | 2048 | 1 | 675 | 1.38e+06 |
| vit_base_patch14_dinov2.lvd142m | 0.9 | pca64 | 64 | 0.646 |  |  |
| vit_base_patch14_dinov2.lvd142m | 0.9 | pca128 | 128 | 0.786 |  |  |
| vit_base_patch14_dinov2.lvd142m | 0.9 | pca256 | 256 | 0.876 |  |  |
| vit_base_patch14_dinov2.lvd142m | 0.9 | native | 768 | 1 | 251 | 1.93e+05 |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 0.9 | pca64 | 64 | 0.527 |  |  |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 0.9 | pca128 | 128 | 0.656 |  |  |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 0.9 | pca256 | 256 | 0.735 |  |  |
| vit_base_patch16_224.augreg_in21k_ft_in1k | 0.9 | native | 768 | 1 | 380 | 2.92e+05 |
| vit_base_patch16_224.mae | 0.9 | pca64 | 64 | 0.371 |  |  |
| vit_base_patch16_224.mae | 0.9 | pca128 | 128 | 0.343 |  |  |
| vit_base_patch16_224.mae | 0.9 | pca256 | 256 | 0.324 |  |  |
| vit_base_patch16_224.mae | 0.9 | native | 768 | 1 | 345 | 2.65e+05 |
| vit_base_patch16_clip_224.openai | 0.9 | pca64 | 64 | 0.626 |  |  |
| vit_base_patch16_clip_224.openai | 0.9 | pca128 | 128 | 0.712 |  |  |
| vit_base_patch16_clip_224.openai | 0.9 | pca256 | 256 | 0.739 |  |  |
| vit_base_patch16_clip_224.openai | 0.9 | native | 512 | 1 | 378 | 1.94e+05 |


Where work is lowest. rises_below: a smaller dimension still reaches the target but needs more work (the interior optimum of P3); ceiling_below: a smaller dimension cannot reach the target because exact search there already loses too many neighbours.

| model | index | target | dims_tested | argmin_dim | smallest_feasible_dim | rises_below | ceiling_below | work_native_over_min |
|---|---|---|---|---|---|---|---|---|
| convnext_base.fb_in22k_ft_in1k | hnsw | 0.8 | 64,128,256,1024 | 256 | 256 | False | True | 1.01 |
| convnext_base.fb_in22k_ft_in1k | hnsw | 0.9 | 64,128,256,1024 | 1024 | 1024 | False | True | 1 |
| resnet50.a1_in1k | hnsw | 0.8 | 64,128,256,2048 | 2048 | 2048 | False | True | 1 |
| resnet50.a1_in1k | hnsw | 0.9 | 64,128,256,2048 | 2048 | 2048 | False | True | 1 |
| vit_base_patch14_dinov2.lvd142m | hnsw | 0.8 | 64,128,256,768 | 256 | 256 | False | True | 3.07 |
| vit_base_patch14_dinov2.lvd142m | hnsw | 0.9 | 64,128,256,768 | 768 | 768 | False | True | 1 |
| vit_base_patch16_224.augreg_in21k_ft_in1k | hnsw | 0.8 | 64,128,256,768 | 768 | 768 | False | True | 1 |
| vit_base_patch16_224.augreg_in21k_ft_in1k | hnsw | 0.9 | 64,128,256,768 | 768 | 768 | False | True | 1 |
| vit_base_patch16_224.mae | hnsw | 0.8 | 64,128,256,768 | 768 | 768 | False | True | 1 |
| vit_base_patch16_224.mae | hnsw | 0.9 | 64,128,256,768 | 768 | 768 | False | True | 1 |
| vit_base_patch16_clip_224.openai | hnsw | 0.8 | 64,128,256,512 | 512 | 512 | False | True | 1 |
| vit_base_patch16_clip_224.openai | hnsw | 0.9 | 64,128,256,512 | 512 | 512 | False | True | 1 |


## P4 and P5: heads at the same dimension, relative to the plain ranking head

| space_id | model | reduction | hub_skew | lid_median | qlid_median | rc_q10 | aniso_meancos | reduction_recall@10 | label_p@10 | hnsw_ndis | hnsw_e2e_ndis |
|---|---|---|---|---|---|---|---|---|---|---|---|
| convnext_base.fb_in22k_ft_in1k__head_hub256 | convnext_base.fb_in22k_ft_in1k | head_hub | 0.685 | 15.4 | 15.1 | 2.21 | 0.00103 | 0.555 | 0.761 | 252 |  |
| convnext_base.fb_in22k_ft_in1k__head_koleo256 | convnext_base.fb_in22k_ft_in1k | head_koleo | 1.23 | 20.1 | 19.8 | 1.75 | -0.000245 | 0.75 | 0.77 | 349 |  |
| convnext_base.fb_in22k_ft_in1k__head_rank256 | convnext_base.fb_in22k_ft_in1k | head_rank | 1.23 | 16.7 | 16.6 | 1.98 | 2.92e-05 | 0.695 | 0.769 | 263 |  |
| convnext_base.fb_in22k_ft_in1k__pca256 | convnext_base.fb_in22k_ft_in1k | pca | 1.84 | 19.7 | 19.5 | 1.76 | -9.89e-05 | 0.801 | 0.771 | 336 |  |
| resnet50.a1_in1k__head_hub256 | resnet50.a1_in1k | head_hub | 0.687 | 29.1 | 28.6 | 1.51 | 0.00443 | 0.576 | 0.41 | 708 |  |
| resnet50.a1_in1k__head_koleo256 | resnet50.a1_in1k | head_koleo | 1.23 | 32 | 31.5 | 1.42 | 0.00269 | 0.62 | 0.415 | 743 |  |
| resnet50.a1_in1k__head_rank256 | resnet50.a1_in1k | head_rank | 1.24 | 28.5 | 28.2 | 1.51 | 0.00434 | 0.638 | 0.414 | 700 |  |
| resnet50.a1_in1k__pca256 | resnet50.a1_in1k | pca | 1.83 | 24.3 | 24.2 | 1.56 | 0.000951 | 0.623 | 0.42 | 647 |  |
| vit_base_patch14_dinov2.lvd142m__head_hub256 | vit_base_patch14_dinov2.lvd142m | head_hub | 0.721 | 12.5 | 12.4 | 2.5 | 0.000642 | 0.595 | 0.81 | 225 |  |
| vit_base_patch14_dinov2.lvd142m__head_koleo256 | vit_base_patch14_dinov2.lvd142m | head_koleo | 1.16 | 16.7 | 16.7 | 1.93 | 0.000449 | 0.79 | 0.822 | 252 |  |
| vit_base_patch14_dinov2.lvd142m__head_rank256 | vit_base_patch14_dinov2.lvd142m | head_rank | 1.15 | 13.6 | 13.5 | 2.23 | 0.000414 | 0.723 | 0.818 | 236 |  |
| vit_base_patch14_dinov2.lvd142m__pca256 | vit_base_patch14_dinov2.lvd142m | pca | 1.67 | 17 | 16.5 | 1.97 | 0.000337 | 0.876 | 0.821 | 246 |  |
| vit_base_patch16_224.augreg_in21k_ft_in1k__head_hub256 | vit_base_patch16_224.augreg_in21k_ft_in1k | head_hub | 0.67 | 22.3 | 21.6 | 1.75 | 0.00245 | 0.596 | 0.748 | 367 |  |
| vit_base_patch16_224.augreg_in21k_ft_in1k__head_koleo256 | vit_base_patch16_224.augreg_in21k_ft_in1k | head_koleo | 1.18 | 27 | 26.1 | 1.49 | 0.000698 | 0.697 | 0.756 | 408 |  |
| vit_base_patch16_224.augreg_in21k_ft_in1k__head_rank256 | vit_base_patch16_224.augreg_in21k_ft_in1k | head_rank | 1.21 | 23.4 | 22.7 | 1.64 | 0.00148 | 0.701 | 0.755 | 378 |  |
| vit_base_patch16_224.augreg_in21k_ft_in1k__pca256 | vit_base_patch16_224.augreg_in21k_ft_in1k | pca | 1.76 | 22.4 | 22 | 1.61 | 0.000499 | 0.735 | 0.771 | 365 |  |
| vit_base_patch16_224.mae__head_hub256 | vit_base_patch16_224.mae | head_hub | 1.47 | 33.7 | 33.5 | 1.79 | 0.00834 | 0.242 | 0.112 | 713 |  |
| vit_base_patch16_224.mae__head_koleo256 | vit_base_patch16_224.mae | head_koleo | 4.46 | 44.5 | 43.9 | 1.35 | 0.00027 | 0.0918 | 0.164 | 1.4e+03 |  |
| vit_base_patch16_224.mae__head_rank256 | vit_base_patch16_224.mae | head_rank | 3.05 | 40.4 | 40.5 | 1.49 | 0.00185 | 0.138 | 0.143 | 1.26e+03 |  |
| vit_base_patch16_224.mae__pca256 | vit_base_patch16_224.mae | pca | 1.45 | 20.8 | 20.9 | 3.24 | -0.00185 | 0.324 | 0.0623 | 467 |  |
| vit_base_patch16_clip_224.openai__head_hub256 | vit_base_patch16_clip_224.openai | head_hub | 0.665 | 33.3 | 32.3 | 1.29 | 0.000594 | 0.49 | 0.732 | 812 |  |
| vit_base_patch16_clip_224.openai__head_koleo256 | vit_base_patch16_clip_224.openai | head_koleo | 0.986 | 33.1 | 32.2 | 1.29 | 0.000799 | 0.517 | 0.745 | 798 |  |
| vit_base_patch16_clip_224.openai__head_rank256 | vit_base_patch16_clip_224.openai | head_rank | 1.04 | 32.1 | 31.2 | 1.31 | 0.000938 | 0.535 | 0.748 | 779 |  |
| vit_base_patch16_clip_224.openai__pca256 | vit_base_patch16_clip_224.openai | pca | 1.63 | 21.1 | 20.5 | 1.65 | 5.57e-05 | 0.739 | 0.8 | 372 |  |


Changes relative to head_rank (n_increase / n_decrease = number of backbones). P4 expects head_hub to lower hub_skew and, if hubs help navigation, to raise hnsw_ndis; P5 expects head_koleo to change qlid_median or rc_q10 and, through them, hnsw_ndis. The pca rows compare PCA with head_rank.

| reduction | metric | n_backbones | mean_change | n_increase | n_decrease |
|---|---|---|---|---|---|
| head_hub | hub_skew | 6 | -0.67 | 0 | 6 |
| head_hub | qlid_median | 6 | -1.51 | 2 | 4 |
| head_hub | rc_q10 | 6 | 0.146 | 5 | 1 |
| head_hub | label_p@10 | 6 | -0.0123 | 0 | 6 |
| head_hub | reduction_recall@10 | 6 | -0.0625 | 1 | 5 |
| head_hub | hnsw_ndis | 6 | -90 | 2 | 4 |
| head_koleo | hub_skew | 6 | 0.221 | 3 | 3 |
| head_koleo | qlid_median | 6 | 2.91 | 6 | 0 |
| head_koleo | rc_q10 | 6 | -0.157 | 0 | 6 |
| head_koleo | label_p@10 | 6 | 0.00401 | 5 | 1 |
| head_koleo | reduction_recall@10 | 6 | 0.00584 | 2 | 4 |
| head_koleo | hnsw_ndis | 6 | 55.4 | 6 | 0 |
| pca | hub_skew | 6 | 0.211 | 5 | 1 |
| pca | qlid_median | 6 | -4.84 | 2 | 4 |
| pca | rc_q10 | 6 | 0.271 | 3 | 3 |
| pca | label_p@10 | 6 | -0.000157 | 5 | 1 |
| pca | reduction_recall@10 | 6 | 0.111 | 5 | 1 |
| pca | hnsw_ndis | 6 | -197 | 2 | 4 |


## P2: text queries against image embeddings (CLIP)

| query_type | n_queries | reaches_target | ndis_at_target | nhops_at_target | ef_at_target | max_recall@10 | qshift_nn | qlid_median | rc_q10 | label_p@10_exact | ndis_ratio_vs_image |
|---|---|---|---|---|---|---|---|---|---|---|---|
| image | 2020 | True | 386 | 26 | 16 | 1 | 0.981 | 22.1 | 1.6 | 0.792 | 1 |
| text | 808 | True | 769 | 73 | 64 | 0.999 | 2.52 | 220 | 1.09 | 0.964 | 1.99 |
