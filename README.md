# WQU Capstone

## Regime-based Pair-Trading Model for Intraday Mean Reversion in Indian Stock Markets

---

### Quick Start (Recommended)

**Step 1: Download Data**

Intraday stock data is available for download:

```
https://www.dropbox.com/scl/fi/yfymt0eh0sp3xw92htdxv/data.zip?rlkey=xfqdzj6aack0eheq66bjpvm5q&st=02njn8rg&dl=1
```

Unzip the file and place the contents in the `data/` folder as shown below.

---

**Step 2: Setup Environment**

Use the provided setup scripts for fast and reliable environment setup:

```bash
# Windows PowerShell
./setup.ps1

# Linux/Mac
bash setup.sh
```

These scripts will create a virtual environment, install dependencies, and prepare your workspace automatically.

---

### Manual Setup

If you prefer manual setup, follow these steps:

1. **Download and extract data:**
    - Download from the link above.
    - Extract to the `data/` folder.

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    ```bash
    source ./venv/Scripts/activate  # On Windows using git bash
    ```

4. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

### Project Structure

```
backtest/
│   backtest.py                 # Backtesting script for pairs
common/
│   utils.py                    # Utility functions
│   plots.py                    # Visualization functions
main.py                         # Main entry point for pair selection and analysis


data/
│   2021/Cash Data April 2021/  # Intraday stock CSV files (1-min OHLCV)
│   2022/Cash Data April 2022/  # Intraday stock CSV files (1-min OHLCV)
│   pair_trading_result.json    # JSON file with clusters and stock pairs
```

---

### Running the Main Pipeline

To run the main analysis and pair selection pipeline:

```bash
python main.py
```

---

### Team Members

- Neeraj Patidar – 8patidarneeraj@gmail.com
- Vishesh Mangla - manglavishesh64@gmail.com
- Manish Kumar Chaudhary - mkumarchaudhary06@gmail.com

---

### Folder Preview
<div style="display: grid; grid-template-columns: 1fr 1fr; align-items: center; column-gap: 20rem;">
  <!-- First image -->
  <img src="https://github.com/user-attachments/assets/0debc9da-c619-4787-a65f-73375234afef" width="400">
  
  <!-- Second image (scaled 2x larger) -->
  <img src="https://github.com/user-attachments/assets/9d339ceb-d2e6-4f3f-ac81-17ba00ebb24a" width="600">
</div>



---

**Note:**  
The setup scripts (`setup.ps1` for Windows, `setup.sh` for Linux/Mac) are the main way to get started. Manual instructions are provided for reference.
