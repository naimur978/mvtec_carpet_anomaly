## Setup

**1. Ensure Python version:**
```
Python 3.14.3
```

**2. Create and activate virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Mac
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Prepare dataset:**
Download `carpet.tar.xz` from [here](https://drive.google.com/file/d/1e0BF8gSs6zflzH2tBUN6vW40UjYv_a8N/view) and place it under the `dataset` folder.

**5. Prepare folder structure:**
After step 4, the project structure should look like:
```
upm_interview_3/
├── dataset/
│   └── carpet.tar.xz
├── notebook.ipynb
├── requirements.txt
├── README.md
└── venv/
```

**6. Run notebook:**
Open `notebook.ipynb` and run all cells. Runtime:
- **CPU:** ~3-4 minutes
- **GPU:** ~2 minutes