"""Melanoma triage demo (Gradio).

Upload a skin-lesion photo → the fine-tuned EfficientNetB0 returns P(malignant), a triage decision and a LIME map.
Preparation matches the notebook: RGB, resized to IMG_SIZE x IMG_SIZE, values 0-255.

    pip install -r requirements.txt
    python app.py

The model comes from the notebook: run it on Kaggle with SAVE_MODEL = True, then copy
melanoma_effnetb0.keras and settings.json from /kaggle/working into models/.

Optional environment variables
    MEL_MODEL         path to the .keras model      (default models/melanoma_effnetb0.keras)
    MEL_DATA_DIR      competition folder with jpeg/train/ → example images in the app (none are stored here)
    MEL_LIME_SAMPLES  perturbed images per LIME map (default 200)
"""
import json
import os
import sys
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries

HERE          = Path(__file__).resolve().parent
MODEL_FILE    = Path(os.environ.get('MEL_MODEL', HERE / 'models' / 'melanoma_effnetb0.keras'))
SETTINGS_FILE = HERE / 'models' / 'settings.json'
SETTINGS      = json.loads(SETTINGS_FILE.read_text()) if SETTINGS_FILE.exists() else {}
IMG_SIZE      = SETTINGS.get('img_size', 224)
THRESHOLD     = SETTINGS.get('threshold', 0.5)
LIME_SAMPLES  = int(os.environ.get('MEL_LIME_SAMPLES', 200))

if not MODEL_FILE.exists():
    sys.exit(f'No model at {MODEL_FILE} → run the notebook with SAVE_MODEL = True and copy the .keras file and settings.json into models/')

import keras  # noqa: E402

MODEL     = keras.models.load_model(MODEL_FILE, compile=False)
EXPLAINER = lime_image.LimeImageExplainer(random_state=42)


def prepare(image):
    return np.asarray(Image.fromarray(np.asarray(image, dtype=np.uint8)).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))


def predict_rgb(images):
    p = MODEL.predict(np.asarray(images, dtype=np.float32), batch_size=32, verbose=0).ravel()
    return np.stack([1 - p, p], axis=1)


def screen(image):
    if image is None:
        return None, 'Upload a skin-lesion photo.', None
    x = prepare(image)
    p = float(predict_rgb(x[None])[0, 1])
    suspicious = p >= THRESHOLD
    decision = (f'### {"Suspicious → see a dermatologist soon" if suspicious else "Low risk → routine skin check"}\n'
                f'P(malignant) = **{p:.3f}** · threshold {THRESHOLD:.3f} (picked on validation, max TPR − FPR)\n\n'
                f'A triage aid that sorts photos for a doctor, not a diagnosis.')
    explanation = EXPLAINER.explain_instance(x, predict_rgb, labels=(1,), hide_color=0,
                                             num_samples=LIME_SAMPLES, batch_size=32, random_seed=42)
    img, mask = explanation.get_image_and_mask(1, positive_only=False, num_features=10, hide_rest=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    axes[0].imshow(x)
    axes[0].set_title(f'Input ({IMG_SIZE} × {IMG_SIZE})')
    axes[1].imshow(mark_boundaries(img / 255.0, mask))
    axes[1].set_title('LIME: green → towards malignant, red → towards benign')
    for ax in axes:
        ax.axis('off')
    fig.tight_layout()
    return {'malignant': p, 'benign': 1 - p}, decision, fig


def examples():
    root = os.environ.get('MEL_DATA_DIR')
    if not root:
        return []
    folder = next((p for p in Path(root).rglob('train') if p.is_dir() and p.parent.name == 'jpeg'), None)
    return [[str(p)] for p in sorted(folder.glob('*.jpg'))[:6]] if folder else []


EXAMPLES = examples()
with gr.Blocks(title='Melanoma triage') as demo:
    gr.Markdown('# Melanoma Triage\n'
                'Upload a skin-lesion photo. The fine-tuned EfficientNetB0 returns the probability of melanoma, '
                'a triage decision and a LIME map of the parts of the photo behind the score.')
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type='numpy', label='Skin-lesion photo', height=380)
            btn = gr.Button('Check', variant='primary')
            if EXAMPLES:
                gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column():
            prob = gr.Label(num_top_classes=2, label='Probability')
            text = gr.Markdown()
            plot = gr.Plot(label='Explanation', show_label=False)
    btn.click(screen, inputs=inp, outputs=[prob, text, plot])

if __name__ == '__main__':
    demo.launch()
