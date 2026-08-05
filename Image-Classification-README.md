# Image Classification — Flower Species (CNN)

A convolutional neural network built with TensorFlow/Keras to classify flower
images into 5 species: **daisy, dandelion, roses, sunflowers, tulips**.

## Results

- **Test accuracy: 84.2%**
- **Test loss: 0.492**

## Dataset

- [Flowers Dataset](https://www.kaggle.com/datasets/rahmasleam/flowers-dataset) (Kaggle)
- 5 classes, images resized to 224×224
- 80/20 train/validation split
- The `dandelion` class was capped at 600 training images (it was
  significantly larger than the other classes) to reduce class imbalance;
  `class_weight="balanced"` was also applied during training on top of this.

## Model architecture

A 5-block CNN, each block: `Conv2D → BatchNormalization → MaxPool2D → Dropout(0.25)`,
with filter counts doubling per block (32 → 64 → 128 → 256 → 512), followed by
`GlobalAveragePooling2D → Dense(128, relu) → Dropout(0.5) → Dense(5, softmax)`.

- Optimizer: Adam
- Loss: sparse categorical crossentropy
- Callbacks: `EarlyStopping` (patience 5, restores best weights),
  `ReduceLROnPlateau` (factor 0.5, patience 3)
- Trained for up to 30 epochs (early stopping typically ends training sooner)

## How to test / run this

This notebook was written and run on **Kaggle** and references Kaggle-specific
paths (`/kaggle/input/...`, `/kaggle/working/...`). To run it, either:

### Option A — Run on Kaggle (recommended, no setup)
1. Go to the [Flowers Dataset page on Kaggle](https://www.kaggle.com/datasets/rahmasleam/flowers-dataset).
2. Click **New Notebook**, then upload/import `flower_classification_cnn.ipynb`.
3. Attach the same dataset as an input (Kaggle does this automatically if you
   forked from the dataset page).
4. Run all cells top to bottom.

### Option B — Run locally or in Colab
The notebook's data paths (`datapath`, `trainpath`, `testpath` in the second
cell) are hardcoded to Kaggle's filesystem layout and **must be edited** first:
1. Download the [Flowers Dataset](https://www.kaggle.com/datasets/rahmasleam/flowers-dataset) locally.
2. Update `datapath` to point to wherever you extracted it.
3. Update `trainpath`/`testpath` to any writable local folder (e.g. `./train/`, `./test/`).
4. Install dependencies:
   ```
   pip install tensorflow numpy pandas matplotlib opencv-python scikit-learn seaborn
   ```
5. Run all cells top to bottom in Jupyter or VS Code.

**Note:** no trained model file (`.h5`/`.keras`) is included in this repo —
running the notebook end-to-end retrains the model from scratch (roughly a
few minutes per epoch on GPU, longer on CPU) rather than loading a saved one.
