## Overlaying Spaces and Practical Applicability
## of Complex Geometries: Code

### Requirements:
TF 2.0, 512GB Memory (in fact, we did not measure memory consumption, but it was trained on a server with 512GB :) ).

### Section to notebook mapping:

5.1 distortion experiment
    -> /hdssm_story/offline_distortion_all_sphere_fix.ipynb
    
5.1 map experiment
    -> /hdssm_story/offline_ranking_all_sphere_fix.ipynb

Our new WLA6 dataset
    -> /datasets/wla6.edges and /datasets/wla6.nodes

5.2 dssm tiny experiment
    -> /hdssm_story/dssm-exps-tiny-more-metrics.ipynb
    
5.2 dssm large experiment
    -> /hdssm_story/dssm-exps-large.ipynb

(there is no data for dssm exp due to privacy)

5.3 synthetic bipartite graph
    -> /hdssm_story/strange_graph_exp.ipynb

graph itself
    -> /datasets/bg_20_700_0.05.edges

### Additional:
different conversion from distance to probability
    -> /hdssm_story/offline_ranking_1_div_(no_exp)_dist_exp.ipynb
    -> /hdssm_story/offline_ranking_1_div_dist_exp.ipynb

### Some hints:
tighten spaces = overlaying spaces (early name)
triple trainable = universal signature (4.1, also early name)
triple_trainable_lx = overlaying space $$O_{l1}, (t = x)$$
triple_trainable_lx_sq = overlaying space $$O_{l2}, (t = x)$$
