# keras-fruit-classifer

Classifying fruits using a Keras multi-class image classification model and Google Open Images. 

**See the accompanying blog post [here](TODO: insert link to blog post)**

## Getting the data

This project uses the `quilt/open_fruit` [Quilt T4](https://github.com/quiltdata/t4) data package. This data package contains ~30,000 images of various fruits (`banana`, `melon`, etc; there are 12 classes overall) included the [Google Open Images](https://storage.googleapis.com/openimages/web/index.html) ontology (specifically, only images with bounding boxes).

The `initial-exploration.ipynb` and `build-dataset.ipynb` notebooks in the `notebooks/` subfolder walk through the process of exploring raw dataset assets and composing the Open Fruits dataset, respectively.

You can download the finished product yourself using the `t4` Python package:

```python
import t4

# download everything, including the raw images and raw metadata
t4.Package.install('quilt/open_fruit', 's3://quilt-example', dest='./')

# alternatively, download just the cropped images and formatted metadata
open_fruits = t4.Package.browse('quilt/open_fruit', 's3://quilt-example', dest='./')
open_fruits['training_data/X_meta.csv'].fetch('X_meta.csv')
open_fruits['images_cropped'].fetch('images_cropped/')
```

## Building the models

The model definition code is available in the form of `.py` and `.ipynb` files in this repository's `models/` and `notebooks/` folders, respectively.

There are two model architectures to choose from: **InceptionV3** and **VGG16**. For both models, we are using the pretrained networks (both trained on ImageNet) and finetuning the models on our new fruit classes. The training process, results, and experiment metadata for both models is tracked with [Comet.ml](https://www.comet.ml)

```python
from comet_ml import Experiment
experiment = Experiment(api_key="YOUR_API_KEY",project="PROJECT_NAME", workspace="WORKSPACE_NAME")
```

To reproduce the results, it's also helpful to have the exact environment details. You can retrieve the details from Comet.ml through the REST API. See the code here:

```python
PY_VERSION=$(python -c "import comet_ml; print(comet_ml.API().get_experiment_system_details('01e427cedce145f8bc69f19ae9fb45bb')['python_version'])")

conda create -n my_test_env python=$PY_VERSION
conda activate my_test_env

python -c "import comet_ml; print('\n'.join(comet_ml.API().get_experiment_installed_packages('01e427cedce145f8bc69f19ae9fb45bb')))" > requirements.txt
pip install -r requirements.txt
```


## Getting the models

Trained model artifacts are available for browsing via Comet.ml and for download via Quilt t4.

To browse the models generated for this demo in an interactive way [click here](https://www.comet.ml/ceceshao1/comet-quilt-example)

To bring the model artifacts to your local machine, download the `quit/open_fruit_models` package:

```python
import t4
t4.Package.install('quilt/open_fruit_models', 's3://quilt-example', dest='./')
```
