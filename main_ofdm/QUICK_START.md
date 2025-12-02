# Wi-SUN Website Generator - Quick Start

## ✅ WEBSITE SUCCESSFULLY GENERATED!

Your comprehensive Wi-SUN analysis website has been created using data from **17 locations** (Location1.csv to Location17.csv).

## 📂 Output Location

```
wisun_website_output/
├── index.html                      ← OPEN THIS IN YOUR BROWSER
├── time_series_analysis.png
├── env_corr_matrix.png
├── link_reciprocity.png
├── hopcount_rpl_rank.png
├── humidity_vs_rsl.png
└── summary_statistics.csv
```

## 🌐 View Your Website

**Method 1: Direct Open**
```bash
xdg-open wisun_website_output/index.html
```
Or just double-click `index.html` in your file manager.

**Method 2: Local Server (Recommended)**
```bash
cd wisun_website_output
python -m http.server 8000
```
Then open: http://localhost:8000

## 📊 Dataset Summary

- **Total Records**: 551
- **Locations**: 17 (Location1 through Location17)
- **Date Range**: 2025-11-21 15:48:38 to 2025-11-21 22:07:23
- **Parameters**: Temperature, Humidity, RSL_in, RSL_out, RPL_rank, Hopcount, ConnectedTotal, DisconnectedTotal

## 🎨 Website Features

### Interactive Dashboard with 4 Tabs:

1. **🏠 Overview Tab**
   - 3D visualization (Temperature × Humidity × Signal Strength)
   - Correlation matrix heatmap

2. **📈 Time Series Tab**
   - All 7 parameters plotted over time
   - Multiple locations displayed with different colors
   - Interactive Plotly charts (zoom, pan, hover)

3. **🔍 Analysis Tab**
   - Link reciprocity analysis (RSL_in vs RSL_out)
   - Signal strength symmetry visualization

4. **💡 Insights Tab**
   - Downloadable visualizations (PNG format)
   - Summary statistics (CSV format)

## 📈 Generated Visualizations

1. **Time Series Analysis** - All parameters across 17 locations
2. **Correlation Matrix** - Environmental & network parameter correlations
3. **Link Reciprocity** - Signal strength analysis with 4 subplots
4. **Network Topology** - Hopcount, RPL rank, connectivity analysis
5. **Environmental Analysis** - Temperature, humidity impact with anomaly detection

## 🔄 Regenerate Website

To regenerate with updated data:

```bash
python generate_website.py
```

The script will automatically:
- Load all Location*.csv files from `main_ofdm/Locations/`
- Generate all visualizations
- Create interactive HTML dashboard
- Save everything to `wisun_website_output/`

## 📋 Field Mappings

Your ThingSpeak data fields are mapped as:

| Field | Parameter | Unit | Description |
|-------|-----------|------|-------------|
| field1 | Temperature | °C | Ambient temperature |
| field2 | Humidity | % | Relative humidity |
| field3 | DisconnectedTotal | - | Disconnected nodes count |
| field4 | RSL_in | dBm | Received Signal Level (incoming) |
| field5 | RSL_out | dBm | Received Signal Level (outgoing) |
| field6 | RPL_rank | - | Routing Protocol rank |
| field7 | Hopcount | - | Network hops |
| field8 | ConnectedTotal | - | Connected nodes count |

## 🎯 Similar to ESW_Website_KernelKrew

Your generated website includes all the features of ESW_Website_KernelKrew plus:
- ✅ Professional gradient design
- ✅ Interactive Plotly charts
- ✅ Multiple location support (17 locations vs 4)
- ✅ Comprehensive data analysis
- ✅ Tabbed navigation
- ✅ Download capabilities
- ✅ Responsive design
- ✅ Summary statistics cards
- ✅ Environmental correlation analysis
- ✅ Network topology visualization
- ✅ Signal strength reciprocity analysis

## 🛠️ Requirements

Already installed:
- pandas
- numpy
- matplotlib
- seaborn
- plotly

## 📝 Notes

- The website is fully self-contained (except Plotly CDN)
- All PNG images are embedded in the output directory
- Interactive charts require internet for Plotly.js CDN
- Works on all modern browsers (Chrome, Firefox, Safari, Edge)

## 🎉 Enjoy Your Dashboard!

Open `wisun_website_output/index.html` and explore your comprehensive Wi-SUN network analysis!

---
*Generated with ❤️ for Wi-SUN Network Analysis*
