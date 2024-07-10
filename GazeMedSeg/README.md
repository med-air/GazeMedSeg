# GazeMedSeg Dataset

We contribute the GazeMedSeg dataset, a extension to the public Kvasir-SEG and NCI-ISBI datasets with gaze annotation for medical image segmentation. The GazeMedSeg contains fixation sequences of 1000 images in Kvasir-SEG and 789 slices in NCI-ISBI. The GazeMedSeg dataset can be downloaded [here](https://drive.google.com/drive/folders/1-38bG_81OsGVCb_trI00GSqfB_shCUQG?usp=sharing).

## Description

The gaze data contains raw fixation sequences from eye-tracking trials. The filtering rules includes removing fixations with a duration of less than 50 ms and outliers outside the images. The gaze dataset includes the following columns:

- **IMAGE:** The image ID corresponding to the original dataset. Items with the same image ID form the fixation sequence for an image.
- **CURRENT_FIX_INDEX:** The chronological order of current fixation in the sequence of each image, taring from 1.
- **CURRENT_FIX_X:** The normalized X-coordinate of current fixation in the image, where the left is 0 and the right is 1.
- **CURRENT_FIX_Y:** The normalized Y-coordinate of current fixation in the image, where the top is 0 and the bottom is 1.
- **CURRENT_FIX_DURATION:** The duration of current fixation point in ms.
- **CURRENT_FIX_PUPIL:** The pupil size of current fixation, which is not used in our experiments.
- **CURRENT_FIX_START:** The start timestamp in ms of the current fixation in the trial for each image.
- **IMAGE_HEIGHT:** The height of the original image.
- **IMAGE_WIDTH:** The width of the original image.

## Gaze Collection

We use [SR Research Experiment Builder](https://www.sr-research.com/experiment-builder/) to program our eye-tracking trials. Our gaze annotation scheme contains two stages:

- When presented with an image, the annotator first roughly scans the image and locates the target objects.
- The annotator is requested to scan the objects thoroughly. Typically, participants start from central areas and then move on to the boundaries, ensuring that all parts of the target are covered.

For more details, please refer to our paper.