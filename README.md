## 3d Mesh Stats

This repository contains 2 scripts that are used to generate data and save plots about model performance on
reconstructions of scenes from the
scannet++ dataset (https://github.com/scannetpp/scannetpp). The scripts compare a predicted
mesh to a ground truth mesh, and generate plots visualizing the distances between points on a mesh and the closest
surface on another mesh. The plots include
f-score vs distance threshold plots and violin plots (https://mode.com/blog/violin-plot-examples). The f-score vs
threshold plots are saved as PNGs, and the data for the f-score each plot is saved as a .npy file.
## Scripts
### make_plots_from_meshes.py
This script generates the plots and .npy files from 2 directories, one with scannet++ ground truth .ply files and 
another with reconstructed .ply files. This script uses an application called mshcompare 
(https://github.com/tomfunkhouser/gaps/tree/master/apps/mshcompare) to save .npy files describing the distances between 
each point on a mesh and the closest surface on another mesh, and makes the plots based off this data. This script uses 
flags to take in and output data.

*   `--scannetpp_data_dir`: Directory with the scanetpp meshes.
*   `--pred_mesh_dir`: Directory with meshes from one mesh reconstructing algorithm.
*   `--output_dir`: Directory in which to save .npy files that describe differences between 2 meshes
*   `--mshcompare_file`: Directory to mshcompare.exe file
*   `--f_score_percentile_from_median`: Percentile of f-score values to show in aggregated f-score plots. Default is 66%.
*   `--f_score_max_distance_cm`: The maximum distance threshold to plot in the f-score plots. Default is 200 cm.

**Example** \
python make_plots_from_meshes.py \
  --scannetpp_data_dir /path/to/scannetpp/meshes \
  --pred_mesh_dir /path/to/reconstructed/meshes \
  --output_dir /path/to/output/files \
  --mshcompare_file /path/to/mshcompare.exe \
  --f_score_percentile_from_median 75 \
  --f_score_max_distance_cm 150

### 3d_mesh_scripts.py
This script generates the plots and .npy files from the .npy files created my mshcompare.exe in 
make_plots_from_meshes.py. It takes in the input and output directories using flags.
*   `--input_dir`: Directory with the input .npy files.
*   `--reconstructions`: Space separated list of .npy file names to run scripts on.
*   `--gt2pred_suffix`: Suffix for files of projections from ground truth to prediction, without .npy suffix. Default is .gt2pred.
*   `--pred2gt_suffix`: Suffix for files of projections from prediction to ground truth, without .npy suffix. Default is .pred2gt.
*   `--f_score_percentile_from_median`: Percentile of f-score values to show in aggregated f-score plots. Default is 66%.
*   `--f_score_max_distance_cm`: The maximum distance threshold to plot in the f-score plots. Default is 200 cm.

**Examples**

python 3d_mesh_scripts.py \
  --input_dir /path/to/input/npy/files \
  --reconstructions scan1 scan2 scan3 \
  --gt2pred_suffix _gt2pred \
  --pred2gt_suffix _pred2gt \
  --f_score_percentile_from_median 75 \
  --f_score_max_distance_cm 100