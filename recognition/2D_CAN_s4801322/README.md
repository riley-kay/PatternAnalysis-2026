# Evaluatiom of 


## 20 epoch unet 3 layer 32 base channels:

Training time: 1570.46 seconds
Peak allocated GPU memory: 1852.89 MiB
Epoch 8/20 | train loss: 0.0304 | val loss: 0.0708 | Prostate Dice: 0.8221 | Mean foreground Dice: 0.8925
  Dice by class: [0.9973, 0.9818, 0.9342, 0.85, 0.8746, 0.8221]

Trainable parameters: 1,949,286
Peak inference GPU memory: 552.78 MiB
Inference time per 2D slice: mean=2.76 ms, median=2.61 ms, p95=3.29 ms
Test images: 540
Per-class test metrics (across the entire test set):
Class 0: Dice=0.9982, IoU=0.9964, target pixels=5072679
Class 1: Dice=0.9847, IoU=0.9699, target pixels=12083597
Class 2: Dice=0.9361, IoU=0.8799, target pixels=2167899
Class 3: Dice=0.9650, IoU=0.9323, target pixels=419651
Class 4: Dice=0.8465, IoU=0.7339, target pixels=102699
Class 5: Dice=0.8495, IoU=0.7384, target pixels=60035
Mean foreground Dice: 0.9164
Prostate Dice (class 5): 0.8495

Average prostate boundary metrics (2-pixel tolerance):
Precision: 0.7421
Recall:    0.7248
F1 score:  0.7237

## 20 epoch batch norm CAN 32 hidden:
Training time: 1195.64 seconds
Peak allocated GPU memory: 1467.79 MiB 
Epoch 18/20 | train loss: 0.0500 | val loss: 0.0714 | Prostate Dice: 0.7897 | Mean foreground Dice: 0.8717
  Dice by class: [0.9956, 0.9789, 0.9247, 0.8495, 0.8157, 0.7897]

Trainable parameters: 75,078
Peak inference GPU memory: 304.41 MiB
Inference time per 2D slice: mean=1.65 ms, median=1.57 ms, p95=2.11 ms
Test images: 540
Per-class test metrics (across the entire test set):
Class 0: Dice=0.9978, IoU=0.9956, target pixels=5072679
Class 1: Dice=0.9822, IoU=0.9650, target pixels=12083597
Class 2: Dice=0.9283, IoU=0.8661, target pixels=2167899
Class 3: Dice=0.9584, IoU=0.9202, target pixels=419651
Class 4: Dice=0.7906, IoU=0.6537, target pixels=102699
Class 5: Dice=0.8157, IoU=0.6888, target pixels=60035
Mean foreground Dice: 0.8950
Prostate Dice (class 5): 0.8157

Average prostate boundary metrics (2-pixel tolerance):
Precision: 0.5946
Recall:    0.6776
F1 score:  0.6196

## 20 epoch batch norm CAN 64 hidden:
Training time: 2521.01 seconds
Peak allocated GPU memory: 2912.47 MiB
Epoch 14/20 | train loss: 0.0394 | val loss: 0.0635 | Prostate Dice: 0.8277 | Mean foreground Dice: 0.8886
  Dice by class: [0.9967, 0.9808, 0.9322, 0.8496, 0.8525, 0.8277]

Trainable parameters: 297,606
Peak inference GPU memory: 594.20 MiB
Inference time per 2D slice: mean=3.28 ms, median=3.21 ms, p95=3.90 ms
Test images: 540
Per-class test metrics (across the entire test set):
Class 0: Dice=0.9980, IoU=0.9960, target pixels=5072679
Class 1: Dice=0.9834, IoU=0.9673, target pixels=12083597
Class 2: Dice=0.9321, IoU=0.8729, target pixels=2167899
Class 3: Dice=0.9633, IoU=0.9292, target pixels=419651
Class 4: Dice=0.8155, IoU=0.6884, target pixels=102699
Class 5: Dice=0.8355, IoU=0.7174, target pixels=60035
Mean foreground Dice: 0.9060
Prostate Dice (class 5): 0.8355

Average prostate boundary metrics (2-pixel tolerance):
Precision: 0.6471
Recall:    0.7161
F1 score:  0.6661

## 20 epoch CAN batch norm 128 hidden:
Trainable parameters: 1,185,030
Training time: 6705.68 seconds
Peak allocated GPU memory: 5809.09 MiB
Epoch 6/20 | train loss: 0.0389 | val loss: 0.0616 | Prostate Dice: 0.8253 | Mean foreground Dice: 0.8890
  Dice by class: [0.9977, 0.9819, 0.9341, 0.8575, 0.846, 0.8253]

Trainable parameters: 1,185,030
Peak inference GPU memory: 1177.38 MiB
Inference time per 2D slice: mean=9.12 ms, median=8.73 ms, p95=10.12 ms
Test images: 540
Per-class test metrics (across the entire test set):
Class 0: Dice=0.9978, IoU=0.9956, target pixels=5072679
Class 1: Dice=0.9837, IoU=0.9679, target pixels=12083597
Class 2: Dice=0.9336, IoU=0.8754, target pixels=2167899
Class 3: Dice=0.9618, IoU=0.9264, target pixels=419651
Class 4: Dice=0.8162, IoU=0.6895, target pixels=102699
Class 5: Dice=0.8479, IoU=0.7360, target pixels=60035
Mean foreground Dice: 0.9086
Prostate Dice (class 5): 0.8479

Average prostate boundary metrics (2-pixel tolerance):
Precision: 0.6505
Recall:    0.7137
F1 score:  0.6675

The main limitation for the base 2D U-net is that downsampling loses spatial detail, even though skip connections recover some of it.

Classes:
0 - background
1 - body outline
2 - bone
3 - bladder
4 - rectum
5 - prostate


•A title, a description of the algorithm and the problem that it solves (approximately a paragraph),
how it works in a paragraph, and a figure/visualisation.
## Feasability Report
### User Need, Scope
The proposed system is a medical imaging segmentation tool intended to assist by generating
anatomical contours for review. The model will segment 2D HipMRI slices from the keras slices data
dataset into classes, including bladder, bone, rectum, body outline and prostate.
The primary engineering challenge is determining whether high overall segmentation accuracy trans-
lates into reliable anatomical boundaries.
Acceptance criteria:
1. Target Dice score ≥ 0.75 on prostate label on test set for both the U-net and CAN models.
2. Compare segmentation accuracy and boundary errors of the CAN model against the basic U-net model.
3. Analyse 3–5 representative segmentation failures for both models and compare.
4. Measure peak GPU memory usage and runtime to assess computational feasibility of both models.
### Model Choice & Course Concepts
The project will use the preprocessed 2D HipMRI keras slices data dataset and implement a 2D
U-Net using PyTorch.
U-Net uses an encoder-decoder architecture with skip connections. The encoder extracts increas-
ingly abstract spatial features, while the decoder reconstructs pixel-level segmentation predictions.
Skip connections preserve fine spatial information needed for accurate tissue boundaries.
A CAN model uses dilated convolutions to expand the receptive field and incorporate broader spa-
tial context without relying solely on progressively downsampling the input. The CAN model will
be used to improve the segmentation of the data.
The relevant course concepts are convolutional neural networks, feature extraction, downsampling
and upsampling, skip connections, categorical segmentation and loss functions.
### Preliminary Feasibility Evidence
In the 2D HipMRI dataset there are 11460 training images, 660 validation images and 540 test
images. The dimensions of the segmented images match the raw images and will be paired in data
pre-processing to prevent mixing up patient segmented and non segmented data. The segmented
images have 6 classes. In order to run the model to convergence, 50 epochs will be run to validate
each model increasing if the validation loss has not converged. Running 5 epochs took 433 seconds
a bit over 7 minutes with a max GPU memory usage of 940.58MiB.
### Risks, Budget & Fallback
The main technical risks are class imbalance for the U-net, class boundaries, inconsistent MRI
intensities, over-fitting and potential data leakage. Training will use the available GPU resources
on Rangpur and my own 3060ti, with initial experiments restricted to small subsets of the dataset
to minimise computing costs. Full training will follow once the pipeline has been validated. If the
improved architecture proves too computationally expensive or unstable, a smaller network with
fewer feature channels will be evaluated. The standard U-Net will remain available as a working
baseline. The final investigation will determine whether the CAN models different architecture
provides meaningful improvements in boundary accuracy relative to its computational cost.

## Dependencies
• Any dependencies required, including exact versions, and address reproducibility of results.
Python 3.14.7
matplotlib 3.11.2
nibabel  5.4.2
numpy    2.5.2
pillow   12.3.0
torch    2.14.0+cu126
tqdm     4.70.1

• Example inputs, outputs, and visual plots of your algorithm.
• Description of any specific pre-processing used, with references if any. Justify your training,
validation, and testing splits of the data.


## Investigation of the Open Research Dilemma:

– Quantitative benchmarking of your chosen model against your implemented baseline model under
identical evaluation splits.

Metric	           U-Net baseline	   CAN
Prostate Dice	        TBD	           TBD
Prostate IoU	        TBD	           TBD
Mean Dice               TBD	           TBD
Boundary accuracy	    TBD	           TBD
Peak GPU VRAM	        TBD	           TBD
Per-case inference time	TBD	           TBD
Parameter count

– Resource profiling table documenting computational efficiency (peak GPU VRAM, parameter
count, and inference runtime/latency).
Report per-class Sørensen-Dice similarity co-efficients (DSC) and Intersection-over-Union (IoU) on the held-out test split.

Profile peak GPU VRAM and per-case inference time.
## Failure Cases
– In-depth qualitative error autopsies analyzing 3–5 representative failure cases from the dataset
(diagnosing root triggers and failure modes).
1. 
2. 
3. 

## Conclusion about model
– Actionable engineering recommendation and trade-off synthesis for your project manager.

## Artificial Intelligence Usage Disclosure
ChatGPT was used for debugging syntax, Gemini for concept lookup.
• What specific tasks AI assisted with (e.g., scaffolding data loading functions, refactoring plotting
scripts).
• How you audited, verified, and empirically tested the AI-suggested code.

## References
1. 
2. OpenAI. (2026). ChatGPT (Mar 14 version) [Large language model]. https://chat.openai.com/chat