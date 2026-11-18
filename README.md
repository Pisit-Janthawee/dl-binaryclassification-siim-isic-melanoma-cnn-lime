# SIIM-ISIC Melanoma Classification: CNN + LIME

## Objective

- **Problem** → melanoma is the deadliest skin cancer but is curable when caught early; dermatologists see far more harmless moles than melanomas (1 in 57 images here), and waiting lists are long
- **Why this repo** → rank skin-lesion photos by P(malignant) with a fine-tuned CNN, judge it with ROC-AUC instead of the misleading 98 % accuracy, and check with LIME that it looks at the lesion, not at hair, rulers or the camera
- **Real-world use** → triage in teledermatology: suspicious photos reach the dermatologist first, low-risk ones wait; the LIME map shows *where* the model looked

## Highlights

**Question:** why is accuracy useless here?

**Answer:** only 584 of 33,126 images (1.76 %) are malignant. A model that always says "benign" is 98 % accurate and finds no melanoma → the model is judged by ROC-AUC, and the threshold is chosen on validation.

**Question:** who is most at risk?

**Answer:** age matters most: the malignant share rises from about 1 % under 50 to about 9 % at 80+. Small groups get wide Wilson intervals, so the notebook shows the uncertainty next to every rate.

| Group | Malignant share |
|---|---|
| All images | 1.76 % |
| Under 50 | ~1 % |
| 80 and older | ~9 % |

**Question:** can the model cheat by recognising the patient?

**Answer:** yes, if the same patient is on both sides of the split (2,056 patients, median 12 photos each): skin tone, camera and lighting give the patient away. The split is grouped by patient, and the notebook stops if a patient appears twice.

## [1/10] Overview

- **Data** → [SIIM-ISIC Melanoma Classification](https://www.kaggle.com/competitions/siim-isic-melanoma-classification)
- **Task** → dermoscopic image → benign (0) / malignant (1)
- **Metric** → ROC-AUC
- **Notebook** → `melanoma_classification_cnn_lime.ipynb`
- **Backup** → `arch/` (original version)

## [2/10] Run on Kaggle

- **Accelerator** → GPU
- **Internet** → On (ImageNet weights, `pip install lime`)
- **Settings** → top cell of the notebook, documented in its note
- **Safety** → every write is off by default; the notebook only previews
  - `WRITE_CACHE = False` → images are not cached to `.npz`
  - `SAVE_MODEL = False` → model file and `settings.json` are not written
  - `LAUNCH_APP = False` → the Gradio app is not started
  - `SAVE_SUBMISSION = False` → predicts `SUBMISSION_PREVIEW` test images, writes nothing
- **Lighter run** → lower `IMG_SIZE`, `N_BENIGN` (e.g. 1000), epochs or `LIME_SAMPLES`
- **App on Kaggle** → `LAUNCH_APP = True` prints a temporary public Gradio link
- **More data** → raise `N_BENIGN` (e.g. 8000); memory grows ~150 KB per image at 224 px

## [3/10] Metadata EDA

- **Scope** → full train set (33,126 images), not the modeling sample → real rates
- **Class imbalance** → 584 malignant (1.76%)
  - always predicting "benign" scores ~98% accuracy → accuracy is misleading → ROC-AUC
- **Diagnosis vs target** → `diagnosis` is known only after biopsy; every `melanoma` is malignant
  - leakage → never a model input
- **Malignancy rate by site / sex / age** → bar = share of malignant images, dashed line = overall rate
  - error bar = 95% Wilson confidence interval → small groups have wide intervals → may be noise
  - Wilson, part by part → the $z^2/2n$ term pulls the centre towards 50%, so a rare rate gets a lopsided range (oral/genital: 4 of 124 = 3.2%, range 1.3-8.0%); the plain $\hat p \pm 1.96\,\text{SE}$ would give 0.1-6.3%, and "0-0%" for a group without melanoma
  - rate rises with age: ~1% under 50 → ~9% at 80+
- **Patient structure** → 2,056 patients, median 12 images each, 0% of test patients in train
  - same patient in train and test → model recognises the patient (skin tone, camera, lighting) → inflated scores
  - → our split is grouped by patient

## [4/10] Sampling and split

- **Sampling** → all 584 malignant images + `N_BENIGN` = 5,000 random benign images → 5,584 images
  - 6× lighter than the full set, 3.5× more images than a 1,000-benign sample
  - evaluation on this sample is ~10.5% malignant vs 1.76% in real data
- **Split** → `StratifiedGroupKFold`: each patient in exactly one split, similar class ratio
  - test ≈ 1/7 of patients → validation ≈ 1/6 of the rest → ~70 / 15 / 15
  - train 3,986 (416 malignant) · validation 797 (83) · test 801 (85)
  - the notebook stops with an assertion if a patient appears in two splits

## [5/10] Images

- **Loading**
  - **Draft mode** → JPEGs decoded at 1/2, 1/4 or 1/8 scale (originals up to ~6000×4000 px)
  - **Threads** → `N_JOBS` images decoded in parallel
  - **Cache** → `.npz` in `WORK_DIR` when `WRITE_CACHE = True`; re-runs load it if the image list matches
- **Original resolution vs target** → resolution ≈ clinic / camera
  - if it tracks the target, the CNN may learn the acquisition source (borders, blur, colour) instead of the lesion
  - only the JPEG header is read → cheap
- **Colour and contrast** → per-image mean R / G / B and pixel std, as densities per class
  - each class normalised separately → class sizes don't matter
  - differences = colour cues for the CNN, or camera differences (see resolution table)

## [6/10] Hair removal and augmentation

- **Hair removal** → hair covers parts of the lesion and can mislead the CNN
  1. **Grayscale** → RGB → gray
  2. **Kernel** → 17×17 rectangle (`cv2.MORPH_RECT`); other shapes: `MORPH_ELLIPSE`, `MORPH_CROSS`
  3. **Blackhat** → closing − image → highlights thin dark structures (hair) on lighter skin

     ![blackhat](assets/png/hair_removal_1_blackhat.png)
  4. **Threshold** → binary hair mask

     ![threshold](assets/png/hair_removal_2_threshold.png)
  5. **Inpaint** (Telea) → fills the masked pixels from their neighbours

     ![inpaint](assets/png/hair_removal_3_inpaint.png)
- **Augmentation** → over-sample malignant images in the training split only → no augmented copies of val/test images leak into training
  - `AUG_PER_MALIGNANT` = 2 copies per image → 416 + 832 = 1,248 malignant vs 3,570 benign
  - capped → thousands of copies of 416 lesions would teach the same lesions over and over
  - each new image = one training image + one random method
  - **`transform_image`** → rotate ±10°, zoom ×1.1; `BORDER_REPLICATE` fills corners with edge pixels
  - **`random_fscb`** → random flip, saturation, contrast, brightness
  - **`hair_remove`** → the same lesion without hair
- **References**
  - [SIIM Melanoma Competition: EDA + Augmentations (Andrada)](https://www.kaggle.com/code/andradaolteanu/siim-melanoma-competition-eda-augmentations)
  - [OpenCV morphological transformations](https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html)
  - [OpenCV thresholding](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html)
  - [OpenCV inpainting](https://docs.opencv.org/3.4/df/d3d/tutorial_py_inpainting.html)
  - [OpenCV geometric transformations](https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html)
  - [tf.image](https://www.tensorflow.org/api_docs/python/tf/image)

## [7/10] Modeling

- **Model** → EfficientNetB0 pre-trained on ImageNet + new sigmoid head
  - **Input 0-255** → Keras' EfficientNet rescales internally
  - **Online augmentation** → random flip / rotation / zoom inside the model (training only)
  - **Frozen BatchNorm** → backbone called with `training=False`, also when fine-tuning
  - **Mixed precision** on GPU → float16 compute, float32 output → faster, less memory
- **Class weights** → `n / (2 · n_class)` → benign ≈ 0.67, malignant ≈ 1.93 → closes the remaining gap in the loss
- **Sigmoid, part by part** → $p = 1/(1+e^{-z})$ with $z = b + \sum_j w_j h_j$ over the 1,280 pooled features → $z$ is the log-odds; the squashing never changes the order, so ROC-AUC ignores it; $p$ runs higher than the real-life chance (trained on 26% melanoma + weights) → used as a ranking score, threshold picked on validation
- **Weighted binary cross-entropy, part by part** → $L = -w_y[y\log p + (1-y)\log(1-p)]$ → $y$ switches one term on; the log has no ceiling (says 1% for a melanoma → 4.61); $w_y$ is what matters at 1.8%: without it a lazy model saying "1.8%" for every image gets an average loss of only 0.089 and flags nothing, with it the best lazy guess is 50% (loss 0.693)

  ![lazy model and class weights](assets/png/m_bce_weights.png)
- **[1/2] Head** → backbone frozen, Adam `1e-3`, up to `EPOCHS_HEAD`
- **[2/2] Fine-tune** → last `FINE_TUNE_LAST_N` backbone layers unfrozen, Adam `1e-5`, up to `EPOCHS_FINE_TUNE`
- **Best weights** → early stopping on validation AUC per phase; the head weights are restored if fine-tuning didn't beat them (kept in memory, no checkpoint file)

![EfficientNetB0 architecture](assets/png/melanoma_efficientnet_architecture.png)

Editable source: `assets/drawio/melanoma_efficientnet_architecture.drawio`.

## [8/10] Evaluation and LIME

- **Threshold** → chosen on validation (max Youden's J = TPR − FPR), applied unchanged to test
  - the minus weighs one point of melanomas caught the same as one point of false alarms; a clinic may move the line lower on purpose, since a missed melanoma hurts more
- **ROC-AUC** → $P(p_\text{malignant} > p_\text{benign})$ → only the order of the scores counts → independent of the class ratio and of the class weights → comparable to real data
- **Precision** → depends on the class ratio: test is ~10.6% malignant vs 1.76% in reality → real-world precision is much lower
- **Charts** → ROC, precision-recall, predicted-probability histogram, confusion matrix
- **LIME** → hides parts of the image and checks how much the CNN's answer changes → the parts that change it most are the parts the CNN looks at
  1. **Cut into pieces (superpixels)** → about 50 small pieces of similar colour

     ![superpixels](assets/png/lime_1_superpixels.png)
  2. **Altered copies (perturbation)** → 300 copies, about half of the pieces painted black in each
  3. **Ask the CNN** → P(malignant) for every copy: lesion hidden → P goes down, only skin hidden → P stays

     ![perturbation](assets/png/lime_2_perturbation.png)
  4. **Trust similar copies more** → copies that look almost like the original count more: proximity kernel $\pi_x = \exp(-D^2/2\sigma^2)$, $D = 1 - \sqrt{m/K}$ for $m$ of $K$ pieces shown, $\sigma = 0.25$ (LIME default)

     ![proximity weight](assets/png/m_lime_kernel.png)
  5. **Find the important pieces** → straight-line formula $P \approx w_0 + \sum_k w_k z_k$ → $w_k$ = how much P goes up when piece $k$ is shown → a local surrogate fitted by weighted least squares, $\sum_z \pi_x(z)\,(f(z) - g(z))^2$ + a small Ridge penalty: it only has to match the CNN near this one image, and it can miss pieces that only matter together
  6. **Show** → green = towards malignant, red = towards benign

     ![weights](assets/png/lime_3_weights.png)
  - the pictures use a toy model (P goes up when more lesion can be seen); the notebook does the same steps with the real CNN
  - green → towards malignant, red → towards benign
  - one test image per confusion-matrix cell
  - trustworthy → focuses on the lesion; suspicious → hair, rulers, ink marks, dark corners

## [9/10] Gradio app

- **What it does** → upload a skin-lesion photo → P(malignant), a triage decision ("see a dermatologist soon" / "routine skin check") and a LIME map
- **Same preparation as training** → RGB, resized to `IMG_SIZE` × `IMG_SIZE`, values 0-255
- **Decision** → P(malignant) ≥ the threshold picked on validation (max TPR − FPR), read from `models/settings.json`
- **On Kaggle** → `LAUNCH_APP = True` in the top cell
- **On your computer**
  1. run the notebook on Kaggle with `SAVE_MODEL = True`
  2. download `melanoma_effnetb0.keras` and `settings.json` from `/kaggle/working` into `models/`
  3. `pip install -r requirements.txt` → `python app.py`
- **Example images** → `MEL_DATA_DIR=<competition folder> python app.py` adds the first training photos as examples
- A triage aid that sorts photos for a doctor, not a diagnosis

| File | What it is |
|---|---|
| `melanoma_classification_cnn_lime.ipynb` | the whole project, Kaggle-ready |
| `app.py`, `requirements.txt` | Gradio triage app |
| `models/` | trained model + threshold (after you run the notebook) |
| `assets/png/` · `assets/drawio/` | figures (PNG) · editable draw.io diagrams |
| `arch/` | first version of this project |

## [10/10] Next steps

- **All the data** → all 32,542 benign images from a pre-resized (256×256) copy of the JPEGs → fast loading; external malignant images (e.g. ISIC 2019)
- **Metadata** → combine age / sex / site with image features (rates differ by group)
- **Shortcut check** → predictions shouldn't follow image resolution / acquisition source
- **Bigger run** → higher resolution, k-fold ensemble, test-time augmentation
