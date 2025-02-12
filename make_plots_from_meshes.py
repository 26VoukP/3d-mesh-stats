import os
import subprocess
from absl import flags, app
import mesh_stats_plotter

_SCANNETPP_DATA_DIR = flags.DEFINE_string("scannetpp_data_dir", "./scanetpp_data",
                                          "Directory with the scanetpp data, in the structure originally downloaded"
                                          " from the scannetpp dataset. This dir should contain subdirs with scene "
                                          "names. ")
_PRED_MESH_DIR = flags.DEFINE_string("pred_mesh_dir", "./pred_meshes",
                                     "Directory of .ply files of meshes from one reconstruction algorithm. The mesh"
                                     " filenames should include the name of corresponding scene.")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", ".", "Directory in which to put folders for each scene with all data "
                                                     "relating to that scene.")
_MSHCOMPARE_FILE = flags.DEFINE_string("mshcompare_file", "./mshcompare",
                                       "Location and filename of mshcompare executable file.")
_F_SCORE_PERCENTILE_FROM_MEDIAN = flags.DEFINE_integer("f_score_percentile_from_median", 66,
                                                       "Confidence interval for f-score"
                                                       "plot")
_F_SCORE_MAX_DISTANCE_CM = flags.DEFINE_integer("f_score_max_distance_cm", 200, "Maximum distance threshold, in cm, "
                                                                                "plot")

def main(argv):
    assert len(argv) == 1, f"Unrecognized args {argv[1:]}"
    # Iterate over scenes
    scene_output_dirs = []
    for scene_name in os.listdir(_SCANNETPP_DATA_DIR.value):
        scene_dir = os.path.join(_SCANNETPP_DATA_DIR.value, scene_name)
        gt_mesh = os.path.join(scene_dir, "scans", "mesh_aligned_0.05.ply")
        # Create scene directory
        scene_output_dir = os.path.join(_OUTPUT_DIR.value, scene_name)
        scene_output_dirs.append(os.path.basename(scene_output_dir))
        os.makedirs(scene_output_dir, exist_ok=True)
        # assumes pred_mesh will contain of scene name of scene in scannetpp dataset
        pred_meshes = [os.path.join(_PRED_MESH_DIR.value, f) for f in os.listdir(_PRED_MESH_DIR.value) if scene_name in f]
        for pred_mesh in pred_meshes:
            pred_basename = os.path.splitext(os.path.basename(pred_mesh))[0]
            pred_to_gt_stats = os.path.join(scene_output_dir, pred_basename + ".pred2gt.npy")
            gt_to_pred_stats = os.path.join(scene_output_dir, pred_basename + ".gt2pred.npy")

            # Create stats files
            if not os.path.exists(pred_to_gt_stats):
                print(f"Creating {pred_to_gt_stats}")
                subprocess.run([_MSHCOMPARE_FILE.value, pred_mesh, gt_mesh, pred_to_gt_stats, "-v"])
            if not os.path.exists(gt_to_pred_stats):
                print(f"Creating {gt_to_pred_stats}")
                subprocess.run([_MSHCOMPARE_FILE.value, gt_mesh, pred_mesh, gt_to_pred_stats, "-v"])
    mesh_stats_plotter.make_plots(_OUTPUT_DIR.value, scene_output_dirs, ".gt2pred.npy",
                                  _F_SCORE_PERCENTILE_FROM_MEDIAN.value, _F_SCORE_MAX_DISTANCE_CM.value)

if __name__ == "__main__":
    app.run(main)