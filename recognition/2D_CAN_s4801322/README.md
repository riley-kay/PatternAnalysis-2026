# Evaluatiom of 



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