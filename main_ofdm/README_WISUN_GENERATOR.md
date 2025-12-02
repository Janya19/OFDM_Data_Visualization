# Wi-SUN Website & Data Analysis Generator

A comprehensive Python tool that generates an interactive website similar to ESW_Website_KernelKrew with extensive data analysis and visualizations for Wi-SUN sensor network testing.

## 📋 Features

- **Interactive HTML Dashboard**: Beautiful, responsive website with tabbed navigation
- **Comprehensive Data Analysis**: 
  - Time series analysis for all parameters
  - Correlation matrix with heatmaps
  - Signal strength reciprocity analysis
  - Network topology visualization
  - Environmental impact analysis
  - Anomaly detection
- **Multiple Data Sources**:
  - Load from CSV files (ThingSpeak exports or manual data)
  - Fetch directly from ThingSpeak API
  - Generate example/test data
- **Rich Visualizations**:
  - Static PNG plots (Matplotlib/Seaborn)
  - Interactive Plotly charts
  - 3D scatter plots
  - Downloadable reports

## 📊 Field Mappings

Based on your ThingSpeak channel:

| Field | Parameter | Unit | Description |
|-------|-----------|------|-------------|
| Field 1 | Temperature | °C | Ambient temperature |
| Field 2 | Humidity | % | Relative humidity |
| Field 3 | DisconnectedTotal | - | Number of disconnected nodes |
| Field 4 | RSL_in | dBm | Received Signal Level (incoming) |
| Field 5 | RSL_out | dBm | Received Signal Level (outgoing) |
| Field 6 | RPL_rank | - | Routing Protocol for Low-Power rank |
| Field 7 | Hopcount | - | Number of network hops |
| Field 8 | ConnectedTotal | - | Number of connected nodes |

## 🔧 Installation

### Prerequisites

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn plotly requests
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### Create requirements.txt

```txt
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
requests>=2.31.0
```

## 🚀 Usage

### Method 1: Load from CSV Files

If you've exported data from ThingSpeak or have CSV files:

```bash
python wisun_website_generator.py
```

Choose option **1**, then provide:
- Single file: `location1.csv`
- Multiple files: `location1.csv,location2.csv,location3.csv`
- Directory: `Locations/` (processes all CSV files)

**CSV Format:**
```csv
created_at,field1,field2,field3,field4,field5,field6,field7,field8
2025-11-09T11:34:51+00:00,27.4,55.0,0,-71,-95,939,4,1.57
2025-11-09T11:36:01+00:00,27.5,54.2,0,-68,-93,942,4,2.75
```

### Method 2: Fetch from ThingSpeak API

Directly download data from your ThingSpeak channel:

```bash
python wisun_website_generator.py
```

Choose option **2**, then provide:
- **Channel ID**: Your ThingSpeak channel number
- **Read API Key**: Your channel's Read API key
- **Results**: Number of records to fetch (max 8000)

**Example:**
```
Channel ID: 123456
Read API Key: ABCDEFGHIJKLMNOP
Results: 8000
```

### Method 3: Generate Example Data

Test the generator with synthetic data:

```bash
python wisun_website_generator.py
```

Choose option **3** - generates 100 sample records automatically.

## 📁 Output Structure

The script creates a `wisun_website_output/` directory with:

```
wisun_website_output/
├── index.html                      # Main interactive dashboard
├── summary_statistics.csv          # Statistical summary
├── time_series_analysis.png        # All time series plots
├── env_corr_matrix.png            # Correlation heatmap
├── rsl_analysis.png               # Signal strength analysis
├── hopcount_rpl_rank.png          # Network topology
└── humidity_vs_rsl.png            # Environmental analysis
```

## 🖥️ Python API Usage

Use the generator programmatically:

```python
from wisun_website_generator import WiSUNWebsiteGenerator

# Create generator
generator = WiSUNWebsiteGenerator(output_dir='my_output')

# Option 1: Load from CSV
csv_files = ['location1.csv', 'location2.csv']
if generator.load_data_from_csv(csv_files):
    generator.generate_all()

# Option 2: Fetch from ThingSpeak
if generator.fetch_data_from_thingspeak(
    channel_id='123456',
    api_key='YOUR_API_KEY',
    results=5000
):
    generator.generate_all()

# Generate individual components
generator.generate_summary_statistics()
generator.plot_time_series()
generator.plot_correlation_matrix()
generator.plot_signal_analysis()
generator.plot_network_topology()
generator.plot_environmental_analysis()
```

## 📈 Generated Visualizations

### 1. Interactive Dashboard (index.html)
- **Overview Tab**: 3D visualization + Correlation matrix
- **Time Series Tab**: All parameters over time
- **Advanced Analysis Tab**: Link reciprocity + Network insights
- **Insights & Download Tab**: Summary + Download links

### 2. Static Plots
- **Time Series Analysis**: 7 subplots showing trends
- **Correlation Matrix**: Environmental & network correlations
- **RSL Analysis**: 4 comprehensive signal strength views
- **Network Topology**: Hopcount, RPL rank, connectivity
- **Environmental Analysis**: Temperature/humidity impact + anomaly detection

## 🔍 Analysis Features

### Signal Strength Analysis
- Link reciprocity (RSL_in vs RSL_out)
- Signal distribution histograms
- Temporal signal variations
- Signal vs connectivity correlation

### Network Topology
- RPL rank vs hopcount relationships
- Connectivity trends over time
- Hopcount distribution
- Network quality metrics

### Environmental Impact
- Temperature/humidity correlations
- Environmental impact on signal strength
- Anomaly detection (>2σ)
- Temporal environmental trends

### Statistical Summary
- Mean, median, std deviation
- Min/max ranges
- Data completeness
- Per-location statistics (if applicable)

## 🌐 ThingSpeak Integration

### Exporting Data from ThingSpeak

**Option A: Manual Export**
1. Go to your ThingSpeak channel
2. Click "Data Import/Export"
3. Select "Export recent data"
4. Download CSV for each field or combined

**Option B: API Fetch (Recommended)**
- Use the script's built-in API fetching
- Automatically formats and processes data
- Fetches up to 8000 most recent entries

**API URL Format:**
```
https://api.thingspeak.com/channels/[CHANNEL_ID]/feeds.json?api_key=[API_KEY]&results=8000
```

## 🎨 Customization

### Modify Field Names/Units

Edit the dictionaries in `wisun_website_generator.py`:

```python
FIELD_NAMES = {
    'field1': 'Your_Parameter_Name',
    # ... add more
}

FIELD_UNITS = {
    'field1': 'your_unit',
    # ... add more
}
```

### Change Output Directory

```python
generator = WiSUNWebsiteGenerator(output_dir='custom_output_dir')
```

### Adjust Plot Styles

```python
# Modify these lines in the script
sns.set_style("whitegrid")  # or "darkgrid", "white", etc.
plt.rcParams['figure.figsize'] = (12, 8)  # Adjust size
```

## 📱 Viewing the Dashboard

1. Open `wisun_website_output/index.html` in any modern web browser
2. Navigate between tabs to explore different analyses
3. Hover over Plotly charts for interactive tooltips
4. Download visualizations using the download buttons

**Supported Browsers:**
- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## 🐛 Troubleshooting

### "No data loaded" error
- Check CSV file format matches expected structure
- Ensure CSV has `created_at` column or timestamp as first column
- Verify field names (field1-field8)

### "requests library not installed"
```bash
pip install requests
```

### Plots not displaying in HTML
- Ensure internet connection (for loading Plotly CDN)
- Check browser console for JavaScript errors
- Try a different browser

### Memory issues with large datasets
- Reduce number of results fetched from ThingSpeak
- Process data in batches
- Increase available RAM

## 📊 Example Use Cases

### 1. Campus Wi-SUN Deployment Analysis
```bash
# Analyze multiple measurement locations
python wisun_website_generator.py
# Choose option 1
# Input: main_ofdm/Locations/
```

### 2. Real-time Monitoring Dashboard
```python
# Periodic data fetch and update
import schedule
import time

def update_dashboard():
    generator = WiSUNWebsiteGenerator()
    generator.fetch_data_from_thingspeak('YOUR_CHANNEL', 'YOUR_KEY')
    generator.generate_all()

schedule.every(1).hours.do(update_dashboard)
while True:
    schedule.run_pending()
    time.sleep(60)
```

### 3. Historical Analysis
```python
# Load all historical data
generator = WiSUNWebsiteGenerator()
all_csvs = glob.glob('historical_data/*.csv')
generator.load_data_from_csv(all_csvs)
generator.generate_all()
```

## 🔗 Similar to ESW_Website_KernelKrew

This generator produces websites with:
- ✅ Same visual style and color scheme
- ✅ Interactive Leaflet/Plotly charts
- ✅ Tabbed navigation
- ✅ Comprehensive data analysis
- ✅ Download capabilities
- ✅ Responsive design
- ➕ Enhanced statistical analysis
- ➕ More visualization types
- ➕ Direct ThingSpeak integration

## 📝 License

Free to use for academic and research purposes.

## 🤝 Contributing

Suggestions and improvements welcome!

## 📧 Support

For issues or questions, please refer to the script's docstrings and comments.

---

**Generated with ❤️ for Wi-SUN Network Analysis**
