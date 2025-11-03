# FairScan Quadrilateral model

Used by [FairScan](https://github.com/pynicolas/FairScan)
to approximate the coordinates of a quadrilateral for a document based on a segmentation mask.

## Dataset

The dataset can be found in a separate repository:
[fairscan-dataset](https://github.com/pynicolas/fairscan-dataset/).

## Training
```bash
# 1. Clone the repository
git clone https://github.com/pynicolas/fairscan-quadrilateral
cd fairscan-quadrilateral

# 2. Create a venv
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate.bat on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up the dataset
python setup_dataset.py

# 5. Run the training script
python train.py
```

## License

This repository is released under the GNU GPLv3 license.
See [LICENSE](LICENSE) for details.

