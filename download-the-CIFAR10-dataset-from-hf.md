<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# how do i download the CIFAR10 dataset from hugging face

To download the CIFAR-10 dataset from Hugging Face, you can use the `datasets` library provided by Hugging Face. Here is a step-by-step guide:

## Steps to Download CIFAR-10 Dataset from Hugging Face

1. **Install the `datasets` library** if you haven't already:

```bash
pip install datasets
```

2. **Load the CIFAR-10 dataset** in your Python script or notebook using the `load_dataset` function:

```python
from datasets import load_dataset

# Load the entire CIFAR-10 dataset
dataset = load_dataset('cifar10')

# Alternatively, load specific splits
train_ds = load_dataset('cifar10', split='train')
test_ds = load_dataset('cifar10', split='test')
```

3. **Access the images and labels**:
Each data instance contains an image and its corresponding label. The images are loaded as `PIL.Image.Image` objects.

```python
# Example: Access the first training image and label
image = train_ds[^0]['image']
label = train_ds[^0]['label']
```

4. **Optionally, load only a portion of the dataset** for faster experimentation:

```python
train_ds_small = load_dataset('cifar10', split='train[:5000]')
test_ds_small = load_dataset('cifar10', split='test[:2000]')
```


This method downloads the CIFAR-10 dataset (60,000 32x32 color images in 10 classes) directly from Hugging Face’s dataset repository and caches it locally for future use[^1][^3].

### Additional Notes:

- The dataset is split into 50,000 training images and 10,000 test images.
- The labels correspond to 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
- Images are automatically decoded when accessed via the `"image"` key.
- This dataset is the same as the original CIFAR-10 widely used in computer vision research[^1][^4].

This approach is straightforward, widely used, and integrates seamlessly with Hugging Face’s ecosystem for machine learning workflows.

<div style="text-align: center">⁂</div>

[^1]: https://huggingface.co/datasets/uoft-cs/cifar10

[^2]: https://huggingface.co/spolivin/cnn-cifar10

[^3]: https://theaisummer.com/hugging-face-vit/

[^4]: https://www.tensorflow.org/datasets/catalog/cifar10

[^5]: https://paperswithcode.com/dataset/cifar-10

[^6]: https://huggingface.co/datasets/uoft-cs/cifar10/viewer

[^7]: https://huggingface.co/glopez/cifar-10

[^8]: https://huggingface.co/datasets/graphs-datasets/CIFAR10

[^9]: https://huggingface.co/edadaltocg/vgg16_bn_cifar10

[^10]: https://github.com/huggingface/notebooks/blob/main/examples/image_classification-tf.ipynb

