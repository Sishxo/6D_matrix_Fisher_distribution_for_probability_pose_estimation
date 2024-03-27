# Modified_6D_matrix_Fisher_distribution_for_probability_pose_estimation


This is a repo for our work 《6D Matrix Fisher Distribution for Probability Pose Estimation》.

In our work, we proposed a new kind of 6D fisher distribution which could describe rotation's uncertainty on rotation manifold with 6D rotation representation, maintaining complete continuity. To construct a complete probability density function, we apply a sampling strategy to calculate the normalizing term. Besides, we choose a transformer feature extractor to get global topology information for a better distribution modeling.

Our work has just been submitted to IEEE Signal Processing Letters and is currently under review.

## Setup

### Environment

```
conda env create -f environment.yml
```

### Dataset

The dataset is available at [Pascal3D+](https://cvgl.stanford.edu/projects/pascal3d.html).

You could download it and unzip it at ``$PROJECT_DIR/datasets/``

### Train

```
python main.py --run_name <run_name> --config_file <configs> --gpus <gpu_index>
```

You could supervise the training process by:
```
cd logs/pascal
tensorboard --log_dir=./
```

### Eval
```
python scripts/visualize_pascal3d.py
```
