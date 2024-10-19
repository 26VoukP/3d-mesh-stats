# -*- coding: utf-8 -*-
""" Writes images of graphs representing f_score vs threshold and violinplots of reconstruction distances.
"""

#Import the Libraries
import os
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import dataclasses
from typing import Sequence
from absl import flags
from absl import app
import logging

_INPUT_DIR = flags.DEFINE_string("input_dir", ".", "Directory with the input files.")
_RECONSTRUCTIONS = flags.DEFINE_string("reconstructions", "", "Space separated list of mesh reconstructions to load. They could correspond"
                                           "either to scenes of the same reconstruction method or multiple etc.")
_GT2PRED_SUFFIX = flags.DEFINE_string("gt2pred_suffix", ".gt2pred", "Suffix for files of projections from ground truth to "
                                                                    "prediction, without .npy suffix")
_PRED2GT_SUFFIX = flags.DEFINE_string("pred2gt_suffix", ".pred2gt", "Suffix for files of projections from prediction to "
                                                                    "ground truth, without .npy suffix")

# start at .5cm, lots of small bins, a few large bins at the end up to 10m
DISTANCE_FAKE_DATA_BINS_M = np.array([0.0] + np.geomspace(.1, 10_00, num=40).tolist())/100
COLUMNS = [
  'distance',
  'NdotN',
  'type1',
  'id1',
  'category1',
  'curvature1',
  'boundary1',
  'position1.x',
  'position1.y',
  'position1.z',
  'normal1.x',
  'normal1.y',
  'normal1.z',
  'type2',
  'id2',
  'category2',
  'curvature2',
  'boundary2',
  'position2.x',
  'position2.y',
  'position2.z',
  'normal2.x',
  'normal2.y',
  'normal2.z'
]


def calc_angle_normals(scene_scan_df: pd.DataFrame) -> np.ndarray:
  """Gets the angles between corresponding vertex normals.

  Args:
    scene_scan_df: a pandas DataFrame of two corresponding meshes with
    associated points. It must contain columns normal{1, 2}.{x, y, z}.
  Returns:
    1d array of angles between the corresponding normals, in degrees.
  """
  x1 = scene_scan_df['normal1.x']
  y1 = scene_scan_df['normal1.y']
  z1 = scene_scan_df['normal1.z']
  x2 = scene_scan_df['normal2.x']
  y2 = scene_scan_df['normal2.y']
  z2 = scene_scan_df['normal2.z']
  return np.arccos(np.clip(x1*x2 + y1*y2 + z1*z2, -1, 1)) * 180 / np.pi


@dataclasses.dataclass
class Reconstruction_Info:
  name: str
  gt2pred_file: str
  pred2gt_file: str

@dataclasses.dataclass
class Reconstruction_Data:
  """Class representing 1 scene reconstructed with a particular method.

  Attributes:
    name: str name of the scene.
    gt2pred: pd.DataFrame of ground truth to predicted points.
    pred2gt: pd.DataFrame of predicted to ground truth points.
  """
  name: str
  gt2pred: pd.DataFrame
  pred2gt: pd.DataFrame

  @staticmethod
  def load(information_instance: Reconstruction_Info):
    """Loads a scene from the given files.

    Args:
      information_instance: Scene_Info instance with appropriate information
    """
    gt2pred = pd.DataFrame(
        np.load(information_instance.gt2pred_file), columns=COLUMNS)
    pred2gt = pd.DataFrame(
        np.load(information_instance.pred2gt_file), columns=COLUMNS)
    return Reconstruction_Data(
        name=information_instance.name, gt2pred=gt2pred, pred2gt=pred2gt)


def precision_recall(reconstruction: Reconstruction_Data, dist_thresholds_m: list) -> tuple[np.ndarray, np.ndarray]:
  """Gets precision, recall at multiple thresholds.

    Args:
      reconstruction: the scene at which to evaluate precision and recall
      dist_thresholds_m: a list of every distance threshold value in meters

    Returns:
      A tuple of ndarrays with precision and recall, respectively
  """
  vectorized_thresholds = np.expand_dims(dist_thresholds_m, -1)

  true_pos = np.sum(np.expand_dims(reconstruction.gt2pred['distance'], 0) <= vectorized_thresholds, axis=-1)
  recall = true_pos / len(reconstruction.gt2pred) # 99 x 1/

  # pred_to_gt_filtered = pred_to_gt[pred_to_gt['boundary2'] == b'0']
  pred_to_gt_filtered = reconstruction.pred2gt[reconstruction.pred2gt['boundary2'] == 0.0] # changed because cols are now floats
  true_pos = np.sum(np.expand_dims(pred_to_gt_filtered['distance'],0) <= vectorized_thresholds, axis=-1)
  precision = true_pos / len(pred_to_gt_filtered) # 99 x 1
  return precision, recall

def fscore(reconstruction: Reconstruction_Data, distances_m: np.ndarray) -> np.ndarray:
  """Gets fscore values at a specified group of distance thresholds

    Args:
      reconstruction: the scene at which to evaluate precision and recall
      dist_thresholds_m: a list of every distance threshold value in meters

    Returns:
      A tuple of ndarrays with precision and recall, respectively
  """
  precision, recall = precision_recall(reconstruction, distances_m)
  return 2/(1/precision + 1/recall)

def plot_fscore_and_confidence(
    reconstructions : Sequence[Reconstruction_Info], percent_conf: int, method_name: str, max_dist_thresh_cm: int
  ):
  """Plots fscore on y-axis and distance thresholds on x-axis

    Args:
      reconstructions: information for each scene to be aggregated in the plot
      percent_conf: the confidence interval to plot
      method_name: name of the method whose scans are being plotted
      max_dist_thresh_cm: the maximum distance threshold in centimeters
  """
  logging.info("Plotting f-score for reconstructions %s", ",".join([r.name for r in reconstructions]))
  distances_cm = np.arange(1, max_dist_thresh_cm, 2)
  # f_score_lists is a list of lists of 99 F-score values (distance threshold 0-50 cm)
  # creates a numpy array from f_score_lists with the scenes on axis 0
  f_score_scene_dist = np.vstack(
    [fscore(Reconstruction_Data.load(s), distances_cm / 100)
     for s in reconstructions]
  )
  median_arr = np.median(f_score_scene_dist, axis=0) # takes median across scenes
  low_ci = np.percentile(f_score_scene_dist, percent_conf, axis=0) # takes lower ci across scenes
  high_ci = np.percentile(f_score_scene_dist, 100-percent_conf, axis=0) # takes upper ci across scenes
  plt.plot(distances_cm, median_arr, color='blue')
  # colors in space between ci and median
  plt.fill_between(distances_cm, low_ci, high_ci, color='#ADD8E6', alpha=0.75)
  plt.title(method_name + " F-score")
  plt.xlabel("Distance Threshold(CM)")
  plt.ylabel("F-score")
  plt.savefig('fscore_plot.png')

def get_opposite_direction_col(column_name: str) -> str:
  """Gets the column name of the same attribute in the opposite direction

  Args:
    column_name: name of the column to switch

  Returns:
    name of the column in the opposite direction
  """
  for i in range(len(column_name)):
    if column_name[i] == '1':
      return str(column_name[:i] + '2' + column_name[i+1:])
    elif column_name[i] == '2':
      return column_name[:i] + '1' + column_name[i+1:]

def sample_from_implied_distribution(
    original_sample: Sequence[float],
    sample_size: int,
    bins: np.ndarray=DISTANCE_FAKE_DATA_BINS_M
  ) -> np.ndarray:
  """Draws new samples from the implied distribution of a given sample.

  Args:
    original_sample: sequence from which to build the histogram
    sample_size: number of output samples
    bins: histogram bins

  Returns:
    1d array of new samples.
  """
  histogram, bin_edges = np.histogram(original_sample, bins=bins)
  cum_histogram = np.cumsum(histogram)
  cum_probability = cum_histogram/cum_histogram[-1]
  # Generate "query probabilities" from 0 to 1, every e.g. 1/1000th
  new_sample_query_prob = np.arange(0, 1, 1/sample_size) +  1/sample_size/2
  # For each query, find the 2 cum_probability values that bound it,
  # and return a value interpolated between the corresponding bin edges.
  new_sample = np.interp(
      new_sample_query_prob, xp=[0] + cum_probability.tolist(), fp=bin_edges)
  return new_sample

def resample_scene_distances(reconstruction: Reconstruction_Data, sample_size: int=100000
                             ) -> pd.DataFrame:
  """Resamples the distances in the scene to the given size.

  Args:
    reconstruction: the scene to resample
    sample_size: the number of points to sample

  Returns:
    DataFrame with columns: distance, direction({pred2gt | gt2pred})
  """
  logging.info("Resampling %s", reconstruction.name)

  sample_gt2pred = sample_from_implied_distribution(
      reconstruction.gt2pred['distance'], sample_size)
  # Drop tiny values since they would throw off logarithmic plot and they are
  # arbitrary.
  sample_gt2pred = np.maximum(sample_gt2pred, 1e-4)
  sample_gt2pred = pd.DataFrame(sample_gt2pred, columns=['distance(m)'])
  sample_gt2pred['direction'] = "gt2pred"

  sample_pred2gt = sample_from_implied_distribution(reconstruction.pred2gt['distance'], sample_size)
  sample_pred2gt = np.maximum(sample_pred2gt, 1e-4)
  sample_pred2gt = pd.DataFrame(sample_pred2gt, columns=['distance(m)'])
  sample_pred2gt['direction'] = "pred2gt"

  return pd.concat([sample_gt2pred, sample_pred2gt], ignore_index=True, axis=0)

def create_x_labels_vplot(reconstructions : Sequence[Reconstruction_Info],
                          split_interval : int | list):
  '''Creates a list of labels for each x tick on a violinplot figure

  Args:
    reconstructions: Sequence of names of scenes along with data to load them
    split_interval: how often to split imbetween violin plots

  Returns:
    list of labels for each x tick
  '''

  x_labels = [""]
  for i, scene_info in enumerate(reconstructions):
    if split_interval > 0 and i > 0 and i % split_interval == 0:
      x_labels.append("")
    j = 20
    while j < len(scene_info.name):
      if "\n" not in scene_info.name[j:j+20]:
        scene_info.name = scene_info.name[:j] + "\n" + scene_info.name[j:]
      j += 20
    x_labels.append(scene_info.name)
  return x_labels

# adapt to be save data of previous violin plot
def plot_violin_distance(reconstructions: Sequence[Reconstruction_Info],
                         split_interval: int=0):
  '''Plots each scene in a Sequence of scenes as a violin plot

  Args:
    reconstructions: Sequence of names of scenes along with data to load them
    split_interval: how often to split imbetween violin plots
  '''
  logging.info("Plotting violins for reconstructions %s", ",".join([r.name for r in reconstructions]))

  fig, ax = plt.subplots(figsize=(10, 6))

  x_labels = create_x_labels_vplot(reconstructions, split_interval)


  # Customize the plot (optional)
  plt.xlabel("Scene/Method")
  plt.ylabel("Distance (m)")
  plt.title('Reconstruction Distance')
  x_positions = np.arange(len(x_labels))
  ax.set_xticks(x_positions)
  ax.set_xticklabels(x_labels)
  ax.set_yscale('log')

  for i, scene_information in enumerate(reconstructions):
    if split_interval > 0:
     i += int(i/split_interval)
    i += 1 # +1 is to make first spot empty for readability
    scene_data = Reconstruction_Data.load(scene_information)
    compressed_scene_data = resample_scene_distances(scene_data)
    sns.violinplot(data=compressed_scene_data,
                   x=np.repeat(i, len(compressed_scene_data)), y="distance(m)",
                   hue="direction", split=True, inner="quart",
                   palette={"gt2pred": "skyblue", "pred2gt": "lightcoral"},
                   native_scale=True)

  ax.add_patch(
      Rectangle((0.3, .01), x_positions[-1] + .5, 0, facecolor='none',
                edgecolor='black')
      )
  handles, labels = plt.gca().get_legend_handles_labels()
  handles_to_show = handles[:2]
  labels_to_show = labels[:2]
  plt.legend(handles_to_show, labels_to_show)
  plt.tight_layout()
  plt.savefig('violin_plots.png')

def main(argv):
  assert len(argv) == 1, f"Unrecognized args {argv[1:]}"
  matplotlib.use('Agg')
  # makes a list of my own scene data for testing
  method_1_scans_info = []
  reconstructions = _RECONSTRUCTIONS.value.split(" ")
  for reconstruction in reconstructions:
    method_1_scans_info.append(Reconstruction_Info(name=reconstruction,
                                                   gt2pred_file=os.path.join(_INPUT_DIR.value,
                                                                             reconstruction + _GT2PRED_SUFFIX.value + ".npy"),
                                                   pred2gt_file=os.path.join(_INPUT_DIR.value,
                                                                             reconstruction + _PRED2GT_SUFFIX.value + ".npy"))
                               )
  plot_fscore_and_confidence(method_1_scans_info, 66, "IPhone Scan", 200)
  plot_violin_distance(method_1_scans_info, 0)


# make this a top level executable script
if __name__ == "__main__":
  app.run(main)
