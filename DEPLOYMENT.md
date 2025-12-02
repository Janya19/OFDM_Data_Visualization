# Wi-SUN Network Analysis Dashboard - Deployment Guide

## 🌐 Live Website
Your website is deployed at: **https://janya19.github.io/OFDM_Data_Visualization/**

## 📋 Deployment Steps Completed
✅ Website files copied to repository root
✅ Files committed and pushed to GitHub
✅ Repository: https://github.com/Janya19/OFDM_Data_Visualization

## 🚀 Enable GitHub Pages (Do this now!)

1. Go to: https://github.com/Janya19/OFDM_Data_Visualization/settings/pages
2. Under "Source", select:
   - **Branch:** `main`
   - **Folder:** `/ (root)`
3. Click **"Save"**
4. Wait 1-2 minutes for deployment
5. Visit: https://janya19.github.io/OFDM_Data_Visualization/

## 📁 Deployed Files
- `index.html` - Main dashboard page
- `env_corr_matrix.png` - Correlation matrix visualization
- `humidity_vs_rsl.png` - 3D environmental analysis
- `link_reciprocity.png` - Link reciprocity analysis
- `hopcount_rpl_rank.png` - Network topology
- `time_series_analysis.png` - Time series data
- `summary_statistics.csv` - Statistical analysis

## 🔄 Updating the Website

When you make changes:
```bash
cd /home/jkb/Documents/sem3/esw/website/OFDM_Data_Visualization

# Regenerate website if needed
cd main_ofdm && python generate_website.py

# Copy updated files to root
cp -r main_ofdm/wisun_website_output/* ../

# Commit and push
git add .
git commit -m "Update website"
git push origin main
```

The website will automatically update in 1-2 minutes!

## ✨ Features
- 📊 Interactive Time Series Analysis
- 🎯 3D Visualization (Temperature, Humidity, Signal Strength)
- 🔗 Correlation Matrix Analysis
- 📈 Link Reciprocity Analysis
- 📉 Network Topology Visualization
- 📥 Downloadable visualizations and statistics

## 📊 Dataset
- **Total Records:** 551
- **Locations:** 17
- **Date Range:** 2025-11-21
- **Fields:** Temperature, Humidity, RSL_in, RSL_out, RPL_rank, Hopcount, ConnectedTotal, DisconnectedTotal
