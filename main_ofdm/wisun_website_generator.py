"""
Wi-SUN Testing Website & Data Analysis Generator
================================================
Generates a comprehensive website similar to ESW_Website_KernelKrew
with extensive data analysis and interactive visualizations.

Field Mappings:
- Field 1: Temperature (°C)
- Field 2: Humidity (%)
- Field 3: DisconnectedTotal
- Field 4: RSL_in (dBm)
- Field 5: RSL_out (dBm)
- Field 6: RPL_rank
- Field 7: Hopcount
- Field 8: ConnectedTotal
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
FIELD_NAMES = {
    'field1': 'Temperature',
    'field2': 'Humidity',
    'field3': 'DisconnectedTotal',
    'field4': 'RSL_in',
    'field5': 'RSL_out',
    'field6': 'RPL_rank',
    'field7': 'Hopcount',
    'field8': 'ConnectedTotal'
}

FIELD_UNITS = {
    'field1': '°C',
    'field2': '%',
    'field3': '',
    'field4': 'dBm',
    'field5': 'dBm',
    'field6': '',
    'field7': '',
    'field8': ''
}

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

class WiSUNWebsiteGenerator:
    """Main class for generating Wi-SUN website and analysis"""
    
    def __init__(self, output_dir='wisun_website_output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / 'data'
        self.data_dir.mkdir(exist_ok=True)
        self.df = None
        
    def load_data_from_csv(self, csv_files):
        """Load data from CSV files (can be from ThingSpeak exports)"""
        print("Loading data from CSV files...")
        
        data_frames = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                # Rename columns if needed
                if 'created_at' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['created_at'])
                else:
                    df['timestamp'] = pd.to_datetime(df.iloc[:, 0])
                
                # Extract location from filename
                filename = Path(csv_file).stem
                df['location'] = filename
                
                data_frames.append(df)
                print(f"  ✓ Loaded {filename}: {len(df)} records")
            except Exception as e:
                print(f"  ✗ Error loading {csv_file}: {e}")
        
        if data_frames:
            self.df = pd.concat(data_frames, ignore_index=True)
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            print(f"\n✓ Total records loaded: {len(self.df)}")
            print(f"✓ Total locations: {self.df['location'].nunique()}")
            print(f"✓ Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            return True
        else:
            print("✗ No data loaded!")
            return False
    
    def fetch_data_from_thingspeak(self, channel_id, api_key, results=8000):
        """
        Fetch data directly from ThingSpeak API
        
        Parameters:
        - channel_id: Your ThingSpeak channel ID
        - api_key: Your Read API Key
        - results: Number of results to fetch (max 8000)
        """
        try:
            import requests
            
            print(f"Fetching data from ThingSpeak channel {channel_id}...")
            url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
            params = {
                'api_key': api_key,
                'results': results
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            feeds = data['feeds']
            
            if not feeds:
                print("✗ No data retrieved from ThingSpeak")
                return False
            
            # Convert to DataFrame
            self.df = pd.DataFrame(feeds)
            self.df['timestamp'] = pd.to_datetime(self.df['created_at'])
            self.df['location'] = 'ThingSpeak_Data'
            
            # Convert field columns to numeric
            for i in range(1, 9):
                field_col = f'field{i}'
                if field_col in self.df.columns:
                    self.df[field_col] = pd.to_numeric(self.df[field_col], errors='coerce')
            
            print(f"✓ Fetched {len(self.df)} records from ThingSpeak")
            print(f"✓ Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            return True
            
        except ImportError:
            print("✗ 'requests' library not installed. Run: pip install requests")
            return False
        except Exception as e:
            print(f"✗ Error fetching from ThingSpeak: {e}")
            return False
    
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY STATISTICS")
        print("="*80)
        
        stats_dict = {}
        
        for field in ['field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8']:
            if field in self.df.columns:
                stats_dict[field] = {
                    'name': FIELD_NAMES[field],
                    'unit': FIELD_UNITS[field],
                    'mean': self.df[field].mean(),
                    'median': self.df[field].median(),
                    'std': self.df[field].std(),
                    'min': self.df[field].min(),
                    'max': self.df[field].max(),
                    'count': self.df[field].count()
                }
                
                print(f"\n{FIELD_NAMES[field]} ({FIELD_UNITS[field]}):")
                print(f"  Mean:   {stats_dict[field]['mean']:.2f}")
                print(f"  Median: {stats_dict[field]['median']:.2f}")
                print(f"  Std:    {stats_dict[field]['std']:.2f}")
                print(f"  Range:  {stats_dict[field]['min']:.2f} - {stats_dict[field]['max']:.2f}")
        
        # Save statistics to CSV
        stats_df = pd.DataFrame(stats_dict).T
        stats_df.to_csv(self.output_dir / 'summary_statistics.csv')
        print(f"\n✓ Saved: summary_statistics.csv")
        
        return stats_dict
    
    def plot_time_series(self):
        """Generate time series plots"""
        print("\nGenerating time series plots...")
        
        fields = ['field1', 'field2', 'field4', 'field5', 'field6', 'field7', 'field8']
        fig, axes = plt.subplots(4, 2, figsize=(18, 22))
        axes = axes.flatten()
        
        for idx, field in enumerate(fields):
            if field not in self.df.columns:
                continue
                
            if 'location' in self.df.columns and self.df['location'].nunique() > 1:
                for location in self.df['location'].unique():
                    loc_data = self.df[self.df['location'] == location].sort_values('timestamp')
                    axes[idx].plot(loc_data['timestamp'], loc_data[field], 
                                 label=location, alpha=0.8, linewidth=2)
            else:
                axes[idx].plot(self.df['timestamp'], self.df[field], 
                             color='steelblue', linewidth=2)
            
            axes[idx].set_title(f'{FIELD_NAMES[field]} ({FIELD_UNITS[field]}) Over Time', 
                               fontsize=13, fontweight='bold', pad=10)
            axes[idx].set_xlabel('Time', fontsize=11)
            axes[idx].set_ylabel(f'{FIELD_NAMES[field]} ({FIELD_UNITS[field]})', fontsize=11)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
            if idx == 0 and self.df['location'].nunique() > 1:
                axes[idx].legend(fontsize=9)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'time_series_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: time_series_analysis.png")
        plt.close()
    
    def plot_correlation_matrix(self):
        """Generate correlation heatmap"""
        print("\nGenerating correlation matrix...")
        
        fields = ['field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8']
        available_fields = [f for f in fields if f in self.df.columns]
        
        correlation_matrix = self.df[available_fields].corr()
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   cmap='coolwarm',
                   center=0, 
                   square=True, 
                   linewidths=2,
                   cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
                   xticklabels=[f"{FIELD_NAMES[f]}" for f in available_fields],
                   yticklabels=[f"{FIELD_NAMES[f]}" for f in available_fields],
                   mask=mask,
                   annot_kws={"size": 10, "weight": "bold"})
        
        plt.title('Environmental & Network Correlation Matrix', 
                 fontsize=18, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'env_corr_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: env_corr_matrix.png")
        plt.close()
    
    def plot_signal_analysis(self):
        """Generate comprehensive signal strength analysis"""
        print("\nGenerating signal strength analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # RSL In vs Out scatter
        if 'field4' in self.df.columns and 'field5' in self.df.columns:
            scatter = axes[0, 0].scatter(self.df['field5'], self.df['field4'],
                                        c=self.df['field8'] if 'field8' in self.df.columns else 'steelblue',
                                        cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[0, 0].set_xlabel('RSL_out (dBm)', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('RSL_in (dBm)', fontsize=12, fontweight='bold')
            axes[0, 0].set_title('Link Reciprocity: RSL_in vs RSL_out', 
                                fontsize=14, fontweight='bold', pad=10)
            axes[0, 0].grid(True, alpha=0.3)
            if 'field8' in self.df.columns:
                plt.colorbar(scatter, ax=axes[0, 0], label='ConnectedTotal')
            
            # Add diagonal reference line
            min_val = min(self.df['field4'].min(), self.df['field5'].min())
            max_val = max(self.df['field4'].max(), self.df['field5'].max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 
                           'r--', linewidth=2, alpha=0.7, label='Perfect Reciprocity')
            axes[0, 0].legend()
        
        # RSL distribution
        if 'field4' in self.df.columns and 'field5' in self.df.columns:
            axes[0, 1].hist(self.df['field4'].dropna(), bins=50, alpha=0.7,
                           color='steelblue', edgecolor='black', label='RSL_in', density=True)
            axes[0, 1].hist(self.df['field5'].dropna(), bins=50, alpha=0.7,
                           color='coral', edgecolor='black', label='RSL_out', density=True)
            axes[0, 1].set_xlabel('Signal Strength (dBm)', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Density', fontsize=12, fontweight='bold')
            axes[0, 1].set_title('RSL Distribution', fontsize=14, fontweight='bold', pad=10)
            axes[0, 1].legend(fontsize=11)
            axes[0, 1].grid(True, alpha=0.3)
        
        # RSL over time
        if 'field4' in self.df.columns and 'field5' in self.df.columns:
            axes[1, 0].plot(self.df['timestamp'], self.df['field4'], 
                           label='RSL_in', color='steelblue', linewidth=2, alpha=0.8)
            axes[1, 0].plot(self.df['timestamp'], self.df['field5'], 
                           label='RSL_out', color='coral', linewidth=2, alpha=0.8)
            axes[1, 0].set_xlabel('Time', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Signal Strength (dBm)', fontsize=12, fontweight='bold')
            axes[1, 0].set_title('Signal Strength Over Time', fontsize=14, fontweight='bold', pad=10)
            axes[1, 0].legend(fontsize=11)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Signal vs Connectivity
        if 'field4' in self.df.columns and 'field8' in self.df.columns:
            scatter = axes[1, 1].scatter(self.df['field4'], self.df['field8'],
                                        c=self.df['field7'] if 'field7' in self.df.columns else 'steelblue',
                                        cmap='plasma', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[1, 1].set_xlabel('RSL_in (dBm)', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('ConnectedTotal', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('Signal Strength vs Network Connectivity', 
                                fontsize=14, fontweight='bold', pad=10)
            axes[1, 1].grid(True, alpha=0.3)
            if 'field7' in self.df.columns:
                plt.colorbar(scatter, ax=axes[1, 1], label='Hopcount')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rsl_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: rsl_analysis.png")
        plt.close()
    
    def plot_network_topology(self):
        """Generate network topology analysis"""
        print("\nGenerating network topology analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Hopcount vs RPL Rank
        if 'field6' in self.df.columns and 'field7' in self.df.columns:
            scatter = axes[0, 0].scatter(self.df['field6'], self.df['field7'],
                                        c=self.df['field8'] if 'field8' in self.df.columns else 'steelblue',
                                        cmap='viridis', alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
            axes[0, 0].set_xlabel('RPL Rank', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Hopcount', fontsize=12, fontweight='bold')
            axes[0, 0].set_title('Network Routing: RPL Rank vs Hopcount', 
                                fontsize=14, fontweight='bold', pad=10)
            axes[0, 0].grid(True, alpha=0.3)
            if 'field8' in self.df.columns:
                plt.colorbar(scatter, ax=axes[0, 0], label='ConnectedTotal')
        
        # Hopcount distribution
        if 'field7' in self.df.columns:
            hopcount_counts = self.df['field7'].value_counts().sort_index()
            bars = axes[0, 1].bar(hopcount_counts.index, hopcount_counts.values,
                                 color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
            axes[0, 1].set_xlabel('Hopcount', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
            axes[0, 1].set_title('Hopcount Distribution', fontsize=14, fontweight='bold', pad=10)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', fontsize=10)
        
        # Connectivity over time
        if 'field8' in self.df.columns and 'field3' in self.df.columns:
            ax = axes[1, 0]
            ax.plot(self.df['timestamp'], self.df['field8'],
                   color='green', linewidth=2.5, alpha=0.8, label='Connected')
            ax.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax.set_ylabel('ConnectedTotal', color='green', fontsize=12, fontweight='bold')
            ax.tick_params(axis='y', labelcolor='green')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            ax_twin = ax.twinx()
            ax_twin.plot(self.df['timestamp'], self.df['field3'],
                        color='red', linewidth=2.5, alpha=0.8, label='Disconnected')
            ax_twin.set_ylabel('DisconnectedTotal', color='red', fontsize=12, fontweight='bold')
            ax_twin.tick_params(axis='y', labelcolor='red')
            
            ax.set_title('Network Connectivity Over Time', fontsize=14, fontweight='bold', pad=10)
        
        # RPL Rank over time
        if 'field6' in self.df.columns:
            axes[1, 1].plot(self.df['timestamp'], self.df['field6'],
                           color='purple', linewidth=2, alpha=0.8)
            axes[1, 1].set_xlabel('Time', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('RPL Rank', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('RPL Rank Over Time', fontsize=14, fontweight='bold', pad=10)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'hopcount_rpl_rank.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: hopcount_rpl_rank.png")
        plt.close()
    
    def plot_environmental_analysis(self):
        """Generate environmental factors analysis"""
        print("\nGenerating environmental analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Temperature vs Humidity
        if 'field1' in self.df.columns and 'field2' in self.df.columns:
            scatter = axes[0, 0].scatter(self.df['field1'], self.df['field2'],
                                        c=self.df['field4'] if 'field4' in self.df.columns else 'steelblue',
                                        cmap='RdYlGn', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[0, 0].set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
            axes[0, 0].set_ylabel('Humidity (%)', fontsize=12, fontweight='bold')
            axes[0, 0].set_title('Temperature vs Humidity', fontsize=14, fontweight='bold', pad=10)
            axes[0, 0].grid(True, alpha=0.3)
            if 'field4' in self.df.columns:
                plt.colorbar(scatter, ax=axes[0, 0], label='RSL_in (dBm)')
        
        # Humidity vs RSL
        if 'field2' in self.df.columns and 'field4' in self.df.columns:
            scatter = axes[0, 1].scatter(self.df['field2'], self.df['field4'],
                                        c=self.df['field8'] if 'field8' in self.df.columns else 'steelblue',
                                        cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[0, 1].set_xlabel('Humidity (%)', fontsize=12, fontweight='bold')
            axes[0, 1].set_ylabel('RSL_in (dBm)', fontsize=12, fontweight='bold')
            axes[0, 1].set_title('Humidity Impact on Signal Strength', 
                                fontsize=14, fontweight='bold', pad=10)
            axes[0, 1].grid(True, alpha=0.3)
            if 'field8' in self.df.columns:
                plt.colorbar(scatter, ax=axes[0, 1], label='ConnectedTotal')
        
        # Temperature vs RSL
        if 'field1' in self.df.columns and 'field4' in self.df.columns:
            scatter = axes[0, 2].scatter(self.df['field1'], self.df['field4'],
                                        c=self.df['field8'] if 'field8' in self.df.columns else 'steelblue',
                                        cmap='viridis', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
            axes[0, 2].set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
            axes[0, 2].set_ylabel('RSL_in (dBm)', fontsize=12, fontweight='bold')
            axes[0, 2].set_title('Temperature Impact on Signal Strength', 
                                fontsize=14, fontweight='bold', pad=10)
            axes[0, 2].grid(True, alpha=0.3)
            if 'field8' in self.df.columns:
                plt.colorbar(scatter, ax=axes[0, 2], label='ConnectedTotal')
        
        # Temperature over time
        if 'field1' in self.df.columns:
            axes[1, 0].plot(self.df['timestamp'], self.df['field1'],
                           color='red', linewidth=2, alpha=0.8)
            axes[1, 0].set_xlabel('Time', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            axes[1, 0].set_title('Temperature Over Time', fontsize=14, fontweight='bold', pad=10)
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Humidity over time
        if 'field2' in self.df.columns:
            axes[1, 1].plot(self.df['timestamp'], self.df['field2'],
                           color='blue', linewidth=2, alpha=0.8)
            axes[1, 1].set_xlabel('Time', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Humidity (%)', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('Humidity Over Time', fontsize=14, fontweight='bold', pad=10)
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        
        # Temperature anomaly detection
        if 'field1' in self.df.columns:
            temp_mean = self.df['field1'].mean()
            temp_std = self.df['field1'].std()
            temp_z_scores = np.abs((self.df['field1'] - temp_mean) / temp_std)
            anomalies = temp_z_scores > 2
            
            axes[1, 2].plot(self.df['timestamp'], self.df['field1'],
                           color='green', linewidth=1.5, alpha=0.7, label='Normal')
            axes[1, 2].scatter(self.df[anomalies]['timestamp'], 
                              self.df[anomalies]['field1'],
                              color='red', s=100, marker='x', linewidth=3,
                              label=f'Anomalies (>{2}σ)', zorder=5)
            axes[1, 2].axhline(temp_mean, color='blue', linestyle='--', 
                              linewidth=2, alpha=0.7, label=f'Mean: {temp_mean:.2f}°C')
            axes[1, 2].fill_between(self.df['timestamp'],
                                   temp_mean - 2*temp_std,
                                   temp_mean + 2*temp_std,
                                   alpha=0.2, color='blue', label='±2σ range')
            axes[1, 2].set_xlabel('Time', fontsize=12, fontweight='bold')
            axes[1, 2].set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
            axes[1, 2].set_title('Temperature Anomaly Detection', 
                                fontsize=14, fontweight='bold', pad=10)
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'humidity_vs_rsl.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: humidity_vs_rsl.png")
        plt.close()
    
    def generate_interactive_html(self, stats_dict):
        """Generate interactive HTML dashboard with embedded Plotly charts"""
        print("\nGenerating interactive HTML dashboard...")
        
        # Create Plotly figures
        figs = {}
        
        # Time series plot
        fig_ts = make_subplots(
            rows=4, cols=2,
            subplot_titles=[f"{FIELD_NAMES[f]} ({FIELD_UNITS[f]})" 
                          for f in ['field1', 'field2', 'field4', 'field5', 'field6', 'field7', 'field8']],
            vertical_spacing=0.08
        )
        
        fields_to_plot = ['field1', 'field2', 'field4', 'field5', 'field6', 'field7', 'field8']
        positions = [(1,1), (1,2), (2,1), (2,2), (3,1), (3,2), (4,1)]
        
        for field, (row, col) in zip(fields_to_plot, positions):
            if field in self.df.columns:
                fig_ts.add_trace(
                    go.Scatter(x=self.df['timestamp'], y=self.df[field],
                              mode='lines', name=FIELD_NAMES[field],
                              line=dict(width=2)),
                    row=row, col=col
                )
        
        fig_ts.update_layout(height=1400, showlegend=False, 
                            title_text="Time Series Analysis - All Fields")
        figs['timeseries'] = fig_ts.to_json()
        
        # 3D scatter plot
        if all(f in self.df.columns for f in ['field1', 'field2', 'field4']):
            fig_3d = go.Figure(data=[go.Scatter3d(
                x=self.df['field1'],
                y=self.df['field2'],
                z=self.df['field4'],
                mode='markers',
                marker=dict(
                    size=4,
                    color=self.df['field8'] if 'field8' in self.df.columns else self.df['field4'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Connected")
                ),
                hovertemplate='Temp: %{x:.2f}°C<br>Humidity: %{y:.2f}%<br>RSL_in: %{z:.2f} dBm<extra></extra>'
            )])
            
            fig_3d.update_layout(
                title='3D Visualization: Temperature, Humidity, and Signal Strength',
                scene=dict(
                    xaxis_title='Temperature (°C)',
                    yaxis_title='Humidity (%)',
                    zaxis_title='RSL_in (dBm)'
                ),
                height=700
            )
            figs['3d'] = fig_3d.to_json()
        
        # Correlation heatmap
        fields = ['field1', 'field2', 'field3', 'field4', 'field5', 'field6', 'field7', 'field8']
        available_fields = [f for f in fields if f in self.df.columns]
        corr_matrix = self.df[available_fields].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=[FIELD_NAMES[f] for f in available_fields],
            y=[FIELD_NAMES[f] for f in available_fields],
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.3f}',
            textfont={"size": 11},
            colorbar=dict(title="Correlation")
        ))
        
        fig_corr.update_layout(
            title='Correlation Matrix',
            height=600
        )
        figs['correlation'] = fig_corr.to_json()
        
        # Signal strength scatter
        if 'field4' in self.df.columns and 'field5' in self.df.columns:
            fig_signal = go.Figure()
            fig_signal.add_trace(go.Scatter(
                x=self.df['field5'],
                y=self.df['field4'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.df['field8'] if 'field8' in self.df.columns else 'steelblue',
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Connected")
                ),
                hovertemplate='RSL_out: %{x:.2f} dBm<br>RSL_in: %{y:.2f} dBm<extra></extra>'
            ))
            
            # Add diagonal reference line
            min_val = min(self.df['field4'].min(), self.df['field5'].min())
            max_val = max(self.df['field4'].max(), self.df['field5'].max())
            fig_signal.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash', width=2),
                name='Perfect Reciprocity'
            ))
            
            fig_signal.update_layout(
                title='Link Reciprocity: RSL_in vs RSL_out',
                xaxis_title='RSL_out (dBm)',
                yaxis_title='RSL_in (dBm)',
                height=600
            )
            figs['link_reciprocity'] = fig_signal.to_json()
        
        # Generate HTML
        html_template = self._get_html_template(stats_dict, figs)
        
        output_file = self.output_dir / 'index.html'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_template)
        
        print(f"✓ Saved: index.html")
        print(f"✓ Open {output_file} in your browser to view the dashboard")
    
    def _get_html_template(self, stats_dict, figs):
        """Get HTML template for the website"""
        
        # Calculate summary stats for display
        total_records = len(self.df)
        date_range = f"{self.df['timestamp'].min().strftime('%Y-%m-%d')} to {self.df['timestamp'].max().strftime('%Y-%m-%d')}"
        
        stats_html = ""
        stat_cards = [
            ('field1', 'Temperature', '🌡️'),
            ('field2', 'Humidity', '💧'),
            ('field4', 'RSL_in', '📡'),
            ('field5', 'RSL_out', '📤'),
            ('field7', 'Hopcount', '🔄'),
            ('field8', 'Connected', '🔗')
        ]
        
        for field, name, icon in stat_cards:
            if field in stats_dict:
                stats_html += f"""
            <div class="stat-card">
                <div class="stat-icon">{icon}</div>
                <div class="stat-title">{name}</div>
                <div class="stat-value">{stats_dict[field]['mean']:.2f}</div>
                <div class="stat-unit">{stats_dict[field]['unit']}</div>
                <div class="stat-range">Range: {stats_dict[field]['min']:.2f} - {stats_dict[field]['max']:.2f}</div>
            </div>
"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wi-SUN Network Analysis Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }}

        header {{
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            text-align: center;
            margin-bottom: 30px;
        }}

        h1 {{
            color: #667eea;
            font-size: 3em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }}

        .subtitle {{
            color: #666;
            font-size: 1.2em;
            margin-top: 10px;
        }}

        .info-banner {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}

        .info-banner h3 {{
            margin-bottom: 10px;
            font-size: 1.5em;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .stat-card {{
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}

        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }}

        .stat-icon {{
            font-size: 3em;
            margin-bottom: 10px;
        }}

        .stat-title {{
            color: #666;
            font-size: 0.9em;
            text-transform: uppercase;
            margin-bottom: 10px;
            font-weight: 600;
        }}

        .stat-value {{
            color: #667eea;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }}

        .stat-unit {{
            color: #999;
            font-size: 1em;
            margin-bottom: 10px;
        }}

        .stat-range {{
            color: #888;
            font-size: 0.85em;
            padding-top: 10px;
            border-top: 2px solid #f0f0f0;
        }}

        .tabs {{
            display: flex;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}

        .tab-button {{
            flex: 1;
            padding: 18px 25px;
            background: #f8f9fa;
            border: none;
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: 600;
            color: #666;
            transition: all 0.3s ease;
            border-bottom: 4px solid transparent;
        }}

        .tab-button:hover {{
            background: #e9ecef;
            color: #333;
        }}

        .tab-button.active {{
            background: white;
            color: #667eea;
            border-bottom: 4px solid #667eea;
        }}

        .tab-content {{
            display: none;
            background: white;
            border-radius: 12px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .tab-content.active {{
            display: block;
        }}

        .chart-container {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}

        .chart-title {{
            font-size: 1.8rem;
            margin-bottom: 20px;
            color: #667eea;
            text-align: center;
            font-weight: bold;
        }}

        .section-title {{
            color: #667eea;
            font-size: 2rem;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}

        .description {{
            color: #666;
            font-size: 1.05rem;
            line-height: 1.8;
            margin-bottom: 25px;
        }}

        .insights-box {{
            background: linear-gradient(135deg, #e3f2fd 0%, #e1bee7 100%);
            padding: 25px;
            border-radius: 12px;
            margin: 20px 0;
            box-shadow: 0 3px 15px rgba(0,0,0,0.1);
        }}

        .insights-box h4 {{
            font-size: 1.3rem;
            margin-bottom: 15px;
            color: #1976d2;
        }}

        .insights-box ul {{
            margin: 0;
            padding-left: 25px;
            color: #0d47a1;
        }}

        .insights-box li {{
            margin-bottom: 10px;
            font-size: 1rem;
            line-height: 1.6;
        }}

        .download-section {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
            text-align: center;
        }}

        .download-btn {{
            display: inline-block;
            padding: 15px 30px;
            margin: 10px;
            border: none;
            border-radius: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            cursor: pointer;
            text-decoration: none;
            font-size: 1rem;
            font-weight: 600;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}

        .download-btn:hover {{
            transform: translateY(-3px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }}

        footer {{
            background: white;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            margin-top: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }}

        footer p {{
            color: #666;
            margin: 10px 0;
        }}

        @media (max-width: 768px) {{
            .tabs {{
                flex-direction: column;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            .stat-card {{
                padding: 20px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌐 Wi-SUN Network Analysis Dashboard</h1>
            <p class="subtitle">Real-time Sensor Data Visualization and Network Performance Analysis</p>
        </header>

        <div class="info-banner">
            <h3>📊 Dataset Overview</h3>
            <p><strong>Total Records:</strong> {total_records:,} | <strong>Date Range:</strong> {date_range}</p>
            <p style="margin-top: 10px; font-size: 0.95em;">
                Temperature • Humidity • Signal Strength • Network Routing • Connectivity
            </p>
        </div>

        <h2 class="section-title">📈 Summary Statistics</h2>
        <div class="stats-grid">
{stats_html}
        </div>

        <div class="tabs">
            <button class="tab-button active" onclick="openTab(event, 'overview-tab')">
                🏠 Overview
            </button>
            <button class="tab-button" onclick="openTab(event, 'timeseries-tab')">
                📈 Time Series
            </button>
            <button class="tab-button" onclick="openTab(event, 'analysis-tab')">
                🔍 Advanced Analysis
            </button>
            <button class="tab-button" onclick="openTab(event, 'insights-tab')">
                💡 Insights & Download
            </button>
        </div>

        <!-- Overview Tab -->
        <div id="overview-tab" class="tab-content active">
            <h2 class="section-title">🎯 3D Visualization</h2>
            <p class="description">
                Explore the relationship between temperature, humidity, and signal strength in 3D space.
                Each point represents a measurement, colored by connectivity level.
            </p>
            <div class="chart-container">
                <div id="chart-3d"></div>
            </div>

            <h2 class="section-title">🔗 Correlation Analysis</h2>
            <p class="description">
                Understand how different network and environmental parameters correlate with each other.
                Stronger correlations are shown in darker colors.
            </p>
            <div class="chart-container">
                <div id="chart-correlation"></div>
            </div>
        </div>

        <!-- Time Series Tab -->
        <div id="timeseries-tab" class="tab-content">
            <h2 class="section-title">📈 Time Series Analysis</h2>
            <p class="description">
                Track how all measured parameters evolve over time. Identify trends, patterns, and anomalies
                in temperature, humidity, signal strength, and network performance metrics.
            </p>
            <div class="chart-container">
                <div id="chart-timeseries"></div>
            </div>
        </div>

        <!-- Advanced Analysis Tab -->
        <div id="analysis-tab" class="tab-content">
            <h2 class="section-title">🔍 Link Reciprocity Analysis</h2>
            <p class="description">
                Analyze the symmetry of wireless links by comparing incoming and outgoing signal strength.
                Points closer to the diagonal line indicate better link symmetry.
            </p>
            <div class="chart-container">
                <div id="chart-link-reciprocity"></div>
            </div>

            <div class="insights-box">
                <h4>💡 Key Network Insights</h4>
                <ul>
                    <li><strong>Signal Quality:</strong> Average RSL_in is {stats_dict.get('field4', {}).get('mean', 0):.2f} dBm</li>
                    <li><strong>Network Hops:</strong> Average hopcount is {stats_dict.get('field7', {}).get('mean', 0):.2f}</li>
                    <li><strong>Connectivity:</strong> Average connected nodes: {stats_dict.get('field8', {}).get('mean', 0):.2f}</li>
                    <li><strong>Environmental:</strong> Temperature range {stats_dict.get('field1', {}).get('min', 0):.1f}°C - {stats_dict.get('field1', {}).get('max', 0):.1f}°C</li>
                </ul>
            </div>
        </div>

        <!-- Insights & Download Tab -->
        <div id="insights-tab" class="tab-content">
            <h2 class="section-title">💡 Analysis Summary</h2>
            
            <div class="insights-box">
                <h4>🌡️ Environmental Observations</h4>
                <ul>
                    <li>Temperature varied between {stats_dict.get('field1', {}).get('min', 0):.2f}°C and {stats_dict.get('field1', {}).get('max', 0):.2f}°C</li>
                    <li>Humidity ranged from {stats_dict.get('field2', {}).get('min', 0):.2f}% to {stats_dict.get('field2', {}).get('max', 0):.2f}%</li>
                    <li>Average environmental conditions: {stats_dict.get('field1', {}).get('mean', 0):.2f}°C, {stats_dict.get('field2', {}).get('mean', 0):.2f}% RH</li>
                </ul>
            </div>

            <div class="insights-box">
                <h4>📡 Signal Strength Analysis</h4>
                <ul>
                    <li>RSL_in (incoming signal): {stats_dict.get('field4', {}).get('min', 0):.2f} to {stats_dict.get('field4', {}).get('max', 0):.2f} dBm</li>
                    <li>RSL_out (outgoing signal): {stats_dict.get('field5', {}).get('min', 0):.2f} to {stats_dict.get('field5', {}).get('max', 0):.2f} dBm</li>
                    <li>Signal strength variations indicate changing propagation conditions</li>
                </ul>
            </div>

            <div class="insights-box">
                <h4>🌐 Network Topology Metrics</h4>
                <ul>
                    <li>Hopcount range: {stats_dict.get('field7', {}).get('min', 0):.0f} to {stats_dict.get('field7', {}).get('max', 0):.0f} hops</li>
                    <li>RPL Rank variation: {stats_dict.get('field6', {}).get('min', 0):.0f} to {stats_dict.get('field6', {}).get('max', 0):.0f}</li>
                    <li>Network demonstrates stable routing with occasional adaptations</li>
                </ul>
            </div>

            <div class="download-section">
                <h3 style="margin-bottom: 20px; color: #667eea;">📥 Download Data & Visualizations</h3>
                <p style="color: #666; margin-bottom: 20px;">Download the complete dataset and analysis results</p>
                <a href="summary_statistics.csv" class="download-btn" download>📊 Summary Statistics (CSV)</a>
                <a href="time_series_analysis.png" class="download-btn" download>📈 Time Series Charts (PNG)</a>
                <a href="env_corr_matrix.png" class="download-btn" download>🔗 Correlation Matrix (PNG)</a>
                <a href="rsl_analysis.png" class="download-btn" download>📡 Signal Analysis (PNG)</a>
            </div>
        </div>

        <footer>
            <p><strong>Wi-SUN Network Analysis Dashboard</strong></p>
            <p style="margin-top: 15px;">
                Comprehensive analysis of sensor network data including environmental monitoring,
                signal strength assessment, and network topology visualization
            </p>
            <p style="margin-top: 15px; color: #999; font-size: 0.9em;">
                Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            </p>
        </footer>
    </div>

    <script>
        function openTab(evt, tabName) {{
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {{
                tabContents[i].classList.remove('active');
            }}
            
            const tabButtons = document.getElementsByClassName('tab-button');
            for (let i = 0; i < tabButtons.length; i++) {{
                tabButtons[i].classList.remove('active');
            }}
            
            document.getElementById(tabName).classList.add('active');
            evt.currentTarget.classList.add('active');
        }}

        // Initialize Plotly charts
        var timeseriesData = {figs.get('timeseries', '{}')};
        if (timeseriesData.data) {{
            Plotly.newPlot('chart-timeseries', timeseriesData.data, timeseriesData.layout, {{responsive: true}});
        }}

        var data3d = {figs.get('3d', '{}')};
        if (data3d.data) {{
            Plotly.newPlot('chart-3d', data3d.data, data3d.layout, {{responsive: true}});
        }}

        var corrData = {figs.get('correlation', '{}')};
        if (corrData.data) {{
            Plotly.newPlot('chart-correlation', corrData.data, corrData.layout, {{responsive: true}});
        }}

        var linkData = {figs.get('link_reciprocity', '{}')};
        if (linkData.data) {{
            Plotly.newPlot('chart-link-reciprocity', linkData.data, linkData.layout, {{responsive: true}});
        }}
    </script>
</body>
</html>"""
        
        return html
    
    def generate_all(self):
        """Generate all visualizations and website"""
        if self.df is None or len(self.df) == 0:
            print("✗ No data available. Please load data first.")
            return
        
        print("\n" + "="*80)
        print("GENERATING WI-SUN WEBSITE & ANALYSIS")
        print("="*80)
        
        # Generate summary statistics
        stats_dict = self.generate_summary_statistics()
        
        # Generate all plots
        self.plot_time_series()
        self.plot_correlation_matrix()
        self.plot_signal_analysis()
        self.plot_network_topology()
        self.plot_environmental_analysis()
        
        # Generate interactive HTML
        self.generate_interactive_html(stats_dict)
        
        print("\n" + "="*80)
        print("✅ ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\n📁 Output directory: {self.output_dir.absolute()}")
        print(f"🌐 Main website: {(self.output_dir / 'index.html').absolute()}")
        print("\nGenerated files:")
        print("  • index.html - Interactive dashboard")
        print("  • summary_statistics.csv - Statistical summary")
        print("  • time_series_analysis.png - Time series plots")
        print("  • env_corr_matrix.png - Correlation matrix")
        print("  • rsl_analysis.png - Signal strength analysis")
        print("  • hopcount_rpl_rank.png - Network topology analysis")
        print("  • humidity_vs_rsl.png - Environmental analysis")
        print("\n💡 Open index.html in your web browser to view the interactive dashboard!")


def main():
    """Main execution function with examples"""
    print("="*80)
    print("Wi-SUN WEBSITE & ANALYSIS GENERATOR")
    print("="*80)
    print("\nThis script generates a comprehensive website similar to ESW_Website_KernelKrew")
    print("with extensive data analysis and interactive visualizations.\n")
    
    # Create generator instance
    generator = WiSUNWebsiteGenerator(output_dir='wisun_website_output')
    
    print("Choose data source:")
    print("1. Load from CSV files (exported from ThingSpeak or other sources)")
    print("2. Fetch directly from ThingSpeak API")
    print("3. Use example/test data")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\nEnter CSV file paths (comma-separated) or directory path:")
        print("Example: location1.csv,location2.csv or Locations/")
        path_input = input("Path: ").strip()
        
        csv_files = []
        if ',' in path_input:
            csv_files = [p.strip() for p in path_input.split(',')]
        else:
            path = Path(path_input)
            if path.is_dir():
                csv_files = list(path.glob('*.csv'))
            elif path.is_file():
                csv_files = [path]
        
        if generator.load_data_from_csv(csv_files):
            generator.generate_all()
    
    elif choice == '2':
        channel_id = input("Enter ThingSpeak Channel ID: ").strip()
        api_key = input("Enter Read API Key: ").strip()
        results = input("Number of results to fetch (default 8000): ").strip() or "8000"
        
        if generator.fetch_data_from_thingspeak(channel_id, api_key, int(results)):
            generator.generate_all()
    
    elif choice == '3':
        print("\nGenerating example data...")
        # Create example data
        dates = pd.date_range(start='2025-11-09 11:00:00', periods=100, freq='2min')
        example_data = pd.DataFrame({
            'timestamp': dates,
            'field1': np.random.normal(25, 2, 100),  # Temperature
            'field2': np.random.normal(60, 5, 100),  # Humidity
            'field3': np.random.randint(0, 3, 100),  # DisconnectedTotal
            'field4': np.random.normal(-85, 5, 100),  # RSL_in
            'field5': np.random.normal(-70, 5, 100),  # RSL_out
            'field6': np.random.randint(900, 1500, 100),  # RPL_rank
            'field7': np.random.choice([3, 4, 5], 100),  # Hopcount
            'field8': np.random.uniform(1, 5, 100),  # ConnectedTotal
            'location': 'Example_Location'
        })
        
        generator.df = example_data
        generator.generate_all()
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
