## Download WESAD from Kaggle

The WESAD Datase was download from kaggle using a Kaggle API token. (See step 2 from any .ipynb file in code folder)

You need a Kaggle API token (`kaggle.json`).

```bash
from google.colab import files
import os

print("Upload your kaggle.json API token:")
uploaded = files.upload()   # select kaggle.json
print("Kaggle credentials configured.")
```

```bash
# ── Download and unzip WESAD ──
!pip install -q kaggle
!kaggle datasets download -d orvile/wesad-wearable-stress-affect-detection-dataset -p /content/wesad_raw
!unzip -q /content/wesad_raw/*.zip -d /content/wesad_dataset
!echo "Done. Dataset structure:"
!ls /content/wesad_dataset
```
