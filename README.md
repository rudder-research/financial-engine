# Financial Engine (PRISM Engine)

A financial data analysis and regime detection engine that fetches data from FRED (Federal Reserve Economic Data) and Yahoo Finance, applies mathematical transformations, and performs market regime analysis.

## Features

- **Multi-source Data Fetching**: Automatically fetch data from FRED and Yahoo Finance APIs
- **Regime Detection**: Identify market regimes using advanced mathematical models
- **Mathematical Lenses**: Apply various mathematical transformations and analysis frameworks
- **VCF (Vector Coherence Framework)**: Advanced mathematical modeling for financial analysis
- **Data Normalization**: PRISM normalization for consistent data processing
- **Visualization**: Built-in visualization capabilities for analysis results

## Project Structure

```
financial-engine/
├── config.py                  # Configuration settings
├── main_data_loader.py        # Main data loading script
├── data_loader.py            # PRISM data loader module
├── data_loader_core.py       # Core data loader implementation
├── total_math_model.py       # Complete mathematical model
├── core/
│   ├── prism_normalization.py
│   └── math/                 # Mathematical frameworks
│       ├── mathematical_lenses.py
│       ├── vcf_*.py          # VCF framework components
│       └── ...
├── regime/
│   └── detector.py           # Regime detection
├── analysis/                 # Analysis modules
├── visualization/            # Visualization tools
├── data_raw/                 # Raw data storage (gitignored)
├── outputs/                  # Analysis outputs (gitignored)
└── registry/                 # Metric registry (gitignored)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create an `env` file in the project root (use `.env.example` as template):

```bash
cp .env.example env
```

Edit the `env` file and add your FRED API key:

```
FRED_API_KEY=your_actual_api_key_here
```

**Get a FRED API Key**: https://fred.stlouisfed.org/docs/api/api_key.html

### 3. Paths (Automatic & Portable)

**The engine is fully portable** - it automatically uses relative paths based on where you place it:

- All data is stored relative to the project folder
- `data_raw/`, `registry/`, `outputs/` folders created automatically
- **Drop the folder anywhere and it works** - no configuration needed

To override and use a custom location, set environment variables:

```bash
export PRISM_ENGINE_BASE_DIR=/path/to/your/data
# or for main_data_loader.py
export FINANCIAL_ENGINE_BASE_DIR=/path/to/your/data
```

## Usage

### Running the Main Data Loader

```bash
python main_data_loader.py
```

This will:
1. Load your FRED API key from the `env` file
2. Create necessary directories
3. Fetch data for all registered metrics (FRED & Yahoo Finance)
4. Build a master panel with all data
5. Apply data imputation (forward/backward fill)
6. Slice data from 1975 onwards
7. Save all data to CSV files

### Using in Google Colab

```python
# Mount Google Drive (if saving to Drive)
from google.colab import drive
drive.mount('/content/drive')

# Clone or upload the repository to any location
%cd /content/your-folder/

# Run the data loader - it will create data folders right here
!python main_data_loader.py

# Or store data in Drive by setting environment variable:
import os
os.environ['FINANCIAL_ENGINE_BASE_DIR'] = '/content/drive/MyDrive/financial_data'
!python main_data_loader.py
```

### Data Sources

The engine fetches the following metrics:

**FRED Data (Economic Indicators)**:
- Interest rates (DGS10, DGS2, DGS3MO, yield curves)
- Inflation (CPI, Core CPI, PPI)
- Employment (Unemployment Rate, Payrolls)
- Economic activity (Industrial Production, Housing Starts)
- Monetary (M2, Fed Balance Sheet)
- Financial conditions (ANFCI, NFCI)

**Yahoo Finance Data (Market Data)**:
- Equity indices (SPY, QQQ, IWM)
- Commodities (BCOM, GLD, SLV, USO)
- Fixed income (TLT, BND, IEF, LQD, HYG)
- Currency (DXY)
- Volatility (VIX)
- Sectors (XLU)

## Security Notes

- **NEVER commit the `env` file** - it contains your API key
- The `.gitignore` is configured to exclude sensitive files
- Use environment variables or Colab secrets for API keys in production

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FRED_API_KEY` | FRED API key for economic data | Required |
| `PRISM_ENGINE_BASE_DIR` | Base directory for data storage | Auto-detected |
| `FINANCIAL_ENGINE_BASE_DIR` | Alternative base directory variable | Auto-detected |

## Development

### Adding New Metrics

Edit `main_data_loader.py` and add entries to the `registry_data` list:

```python
{"key": "metric_name", "source": "fred", "ticker": "FRED_TICKER"},
{"key": "metric_name", "source": "yahoo", "ticker": "YAHOO_SYMBOL"}
```

### Running Tests

```bash
# Test mathematical frameworks
python core/math/test_framework.py
```

## Troubleshooting

### FRED API Errors

- Verify your API key is correct in the `env` file
- Check you haven't exceeded API rate limits
- Ensure you have internet connectivity

### Path Errors

- Make sure directories are created with proper permissions
- Check that the base directory path is accessible
- Verify `PRISM_ENGINE_BASE_DIR` or `FINANCIAL_ENGINE_BASE_DIR` if set

### Import Errors

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (requires Python 3.8+)

## Version

Current version: 1.0.0

## Author

Jason Rudder

## License

See project license file for details.
