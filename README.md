# WQU Capstone

## Regime-based Pair-Trading Model for Intraday Mean Reversion in Indian Stock Markets

---

### Data Source

Intraday stock data is available for download:

```
https://www.dropbox.com/scl/fi/yfymt0eh0sp3xw92htdxv/data.zip?rlkey=xfqdzj6aack0eheq66bjpvm5q&st=02njn8rg&dl=1
```

Unzip the file and place the contents in the `data/` folder as shown below.

---

### Quick Setup

You can use the provided setup scripts for a fast start, or follow the manual instructions below.

#### Manual Setup# WQU Capstone

## Regime-based Pair-Trading Model for Intraday Mean Reversion in Indian Stock Markets

---

### Quick Start (Recommended)

Use the provided setup scripts for fast and reliable environment setup:

```bash
# Windows PowerShell
./setup.ps1

# Linux/Mac
bash setup.sh
```

These scripts will create a virtual environment, install dependencies, and prepare your workspace automatically.

---

### Data Source

Intraday stock data is available for download:

```
https://www.dropbox.com/scl/fi/yfymt0eh0sp3xw92htdxv/data.zip?rlkey=xfqdzj6aack0eheq66bjpvm5q&st=02njn8rg&dl=1
```

Unzip the file and place the contents in the `data/` folder as shown below.

---

### Manual Setup

If you prefer manual setup, follow these steps:

1. **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment:**
    ```bash
    source ./venv/Scripts/activate  # On Windows using git bash
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download and extract data:**
    - Download from the link above.
    - Extract to the `data/` folder.

---

### Project Structure

```
backtest/
│   backtest.py                 # Main script to run backtests on all pairs
│   output/                     # Generated strategy plots, PnL CSVs
common/
│   utils.py                    # Utility functions (load data, preprocessing)
│   plots.py                    # Visualization functions
intraday_strategy/
│   mean_reversion_intraday_strategy.py  # Strategy logic with Kalman filter
│   kalman_filter/
│       kalman_filter.py        # Kalman filter hedge ratio estimation
data/
│   2021/Cash Data April 2021/  # Intraday stock CSV files (1-min OHLCV)
│   2022/Cash Data April 2022/  # Intraday stock CSV files (1-min OHLCV)
│   pair_trading_result.json    # JSON file with clusters and stock pairs
```

---

### Running the Backtest

```bash
python backtest/backtest.py
```

---

### Team Members

- Neeraj Patidar – 8patidarneeraj@gmail.com
- Vishesh Mangla - manglavishesh64@gmail.com
- Manish Kumar Chaudhary - mkumarchaudhary06@gmail.com

---

### Folder Preview
<div style="display: flex; justify-content: spaace-around; gap: 6rem; align-items: center;">
  <!-- First image -->
  <img src="https://github.com/user-attachments/assets/0debc9da-c619-4787-a65f-73375234afef" width="400">
  
  <!-- Second image (scaled 2x larger) -->
  <img src="https://github.com/user-attachments/assets/9d339ceb-d2e6-4f3f-ac81-17ba00ebb24a" width="600">
</div>



---

**Note:**  
The setup scripts (`setup.ps1` for Windows, `setup.sh` for Linux/Mac) are the main way to get started. Manual instructions are provided for reference.# WQU Capstone

## Regime-based Pair-Trading Model for Intraday Mean Reversion in Indian Stock Markets

---

### Quick Start (Recommended)

Use the provided setup scripts for fast and reliable environment setup:

```bash
# Windows PowerShell
./setup.ps1

# Linux/Mac
bash setup.sh
```

These scripts will create a virtual environment, install dependencies, and prepare your workspace automatically.

---

### Data Source

Intraday stock data is available for download:

```
https://www.dropbox.com/scl/fi/yfymt0eh0sp3xw92htdxv/data.zip?rlkey=xfqdzj6aack0eheq66bjpvm5q&st=02njn8rg&dl=1
```

Unzip the file and place the contents in the `data/` folder as shown below.

---

### Manual Setup

If you prefer manual setup, follow these steps:

1. **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment:**
    ```bash
    source ./venv/Scripts/activate  # On Windows using git bash
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download and extract data:**
    - Download from the link above.
    - Extract to the `data/` folder.

---

### Project Structure

```
backtest/
│   backtest.py                 # Main script to run backtests on all pairs
│   output/                     # Generated strategy plots, PnL CSVs
common/
│   utils.py                    # Utility functions (load data, preprocessing)
│   plots.py                    # Visualization functions
intraday_strategy/
│   mean_reversion_intraday_strategy.py  # Strategy logic with Kalman filter
│   kalman_filter/
│       kalman_filter.py        # Kalman filter hedge ratio estimation
data/
│   2021/Cash Data April 2021/  # Intraday stock CSV files (1-min OHLCV)
│   2022/Cash Data April 2022/  # Intraday stock CSV files (1-min OHLCV)
│   pair_trading_result.json    # JSON file with clusters and stock pairs
```

---

### Running the Backtest

```bash
python backtest/backtest.py
```

---

### Team Members

- Neeraj Patidar – 8patidarneeraj@gmail.com
- Vishesh Mangla - manglavishesh64@gmail.com
- Manish Kumar Chaudhary - mkumarchaudhary06@gmail.com

---

### Folder Preview

<img width="409" height="531" alt="image" src="https://github.com/user-attachments/assets/ed795e98-d5f1-42f9-a1bc-37d32e52e54f" />

---

**Note:**  
The setup scripts (`setup.ps1` for Windows, `setup.sh` for Linux/Mac) are the main way to get started. Manual instructions are provided for reference.

1. **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

2. **Activate the virtual environment:**
    ```bash
    source ./venv/Scripts/activate  # On Windows using git bash
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Download and extract data:**
    - Download from the link above.
    - Extract to the `data/` folder.

---

### Project Structure

```
backtest/
│   backtest.py                 # Main script to run backtests on all pairs
│   output/                     # Generated strategy plots, PnL CSVs
common/
│   utils.py                    # Utility functions (load data, preprocessing)
│   plots.py                    # Visualization functions
intraday_strategy/
│   mean_reversion_intraday_strategy.py  # Strategy logic with Kalman filter
│   kalman_filter/
│       kalman_filter.py        # Kalman filter hedge ratio estimation
data/
│   2021/Cash Data April 2021/  # Intraday stock CSV files (1-min OHLCV)
│   2022/Cash Data April 2022/  # Intraday stock CSV files (1-min OHLCV)
│   pair_trading_result.json    # JSON file with clusters and stock pairs
```

---

### Running the Backtest

```bash
python backtest/backtest.py
```

---

### Team Members

- Neeraj Patidar – 8patidarneeraj@gmail.com
- Vishesh Mangla - manglavishesh64@gmail.com
- Manish Kumar Chaudhary - mkumarchaudhary06@gmail.com

---

### Folder Preview

<img width="409" height="531" alt="image" src="https://github.com/user-attachments/assets/ed795e98-d5f1-42f9-a1bc-37d32e52e54f" />

---

**Note:**  
For convenience, you may use the setup scripts provided in the repository to automate environment creation and dependency installation.
