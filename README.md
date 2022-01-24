# News

[1/18/2022] This repository is still getting updated.

# ADG-Pose
ADG-Pose is a method of Automated Dataset Generation for real-world human pose estimation. ADG-Pose enables users to determine the *person scale*, *crowdedness*, and *occlusion density* distributions of the generated datasets, allowing for training environments that better match the distributions of the target application.

![Illustrating ADG-Pose](figures/ADG-Pose.png)

The above figure shows the three main stages of ADG-Pose. First, a high accuracy top-down human pose estimation model is used to label ultra-high resolution images. By utilizing ultra-high resolution images and a top-down approach, we can mitigate potential issues with annotating distant people as the absolute resolution of the person crops will still be large enough to ensure high accuracy. Second, we take the fully annotated ultra-high resolution images and generate semi-random crops from them. These crops are semi-random because we introduce user-defined parameters to ensure the final dataset will match the desired statistics. First, the user can determine the resolution range to take crops at. To better detect distant persons, larger resolution crops can be used and downscaled to the desired input resolution, thus mimicking larger distances. Second, the user can determine the maximum, minimum, and mean number of persons in a crop. This allows for customization of how crowded a scene is. Third, the user can specify the desired average IoU between people in the crop, tweaking the overall level of occlusion in the dataset. After these crops are made and the statistics verified, the resulting images and annotations are synthesized into a new multi-resolution dataset. Additional user-defined parameters include the total size of the dataset, "train/val/test splits", image aspect ratio, and skeleton/validation style, which must be compatible with the top-down model used for annotation.


# Environment Setup

We have used Ubuntu 18.04.5, Python 3.6.9, and NVIDIA GPUs for developing the ADG-Pose. The rest of the requirments is listed in Requirements.txt.

# Usage

When making a new output dataset, set up the folder structure beforehand:
```
- generated_dataset
|  - images
|  |  - train
|  |  - val
|  - annotations
```

A sample folder with this structrue is provided under the name generated_dataset.

1. Open the "Decode.py" File and edit the line 260. Look for the comment and change accordingly. The path to the initial dataset must be determined here.
2. Prepare the json file of bounding boxes.
3. Use the appropriate path to a top-down model for extracting keypoints based on prepared bounding boxes.
3. Run Decode.py to automatically annotate the initial dataset. use --json argument to specify location of bounding box json file. Use --cfg to determine path to top-down model configuration file. The final code should look like this:

```
python Decode.py --json path_to_bbox/bbox.json --cfg path_to_top_down_model/config.yaml
```

5. Run Crop.py to create a cropped dataset from the annotations and high resolution images. Please open the code and make changes according to commenting in the file. For running the "Crop.py" use the following command:
```
python Crop.py
```

# Citation

Will be announced.
