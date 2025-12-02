#!/usr/bin/env python3
"""
ThingSpeak Batch Processor
===========================
Process multiple ThingSpeak channels and generate individual
or combined analysis reports.
"""

from wisun_website_generator import WiSUNWebsiteGenerator
import pandas as pd
from pathlib import Path
import json

class ThingSpeakBatchProcessor:
    """Process multiple ThingSpeak channels"""
    
    def __init__(self, config_file='thingspeak_channels.json'):
        self.config_file = config_file
        self.channels = []
        
    def load_config(self):
        """Load channel configuration from JSON file"""
        if not Path(self.config_file).exists():
            print(f"Config file {self.config_file} not found. Creating template...")
            self.create_template_config()
            return False
        
        with open(self.config_file, 'r') as f:
            config = json.load(f)
            self.channels = config.get('channels', [])
        
        print(f"✓ Loaded {len(self.channels)} channel configurations")
        return True
    
    def create_template_config(self):
        """Create a template configuration file"""
        template = {
            "channels": [
                {
                    "name": "Channel_1",
                    "channel_id": "123456",
                    "api_key": "YOUR_READ_API_KEY_1",
                    "description": "First measurement location",
                    "enabled": True
                },
                {
                    "name": "Channel_2",
                    "channel_id": "234567",
                    "api_key": "YOUR_READ_API_KEY_2",
                    "description": "Second measurement location",
                    "enabled": True
                }
            ],
            "settings": {
                "results_per_channel": 5000,
                "output_base_dir": "batch_output",
                "combine_all": True
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(template, f, indent=4)
        
        print(f"✓ Created template config: {self.config_file}")
        print(f"  Edit this file with your channel IDs and API keys")
    
    def process_individual_channels(self):
        """Process each channel separately"""
        print("\n" + "="*80)
        print("PROCESSING INDIVIDUAL CHANNELS")
        print("="*80)
        
        results = []
        
        for idx, channel in enumerate(self.channels, 1):
            if not channel.get('enabled', True):
                print(f"\n[{idx}] Skipping disabled channel: {channel['name']}")
                continue
            
            print(f"\n[{idx}/{len(self.channels)}] Processing: {channel['name']}")
            print(f"  Channel ID: {channel['channel_id']}")
            print(f"  Description: {channel.get('description', 'N/A')}")
            
            output_dir = f"batch_output/{channel['name']}"
            generator = WiSUNWebsiteGenerator(output_dir=output_dir)
            
            success = generator.fetch_data_from_thingspeak(
                channel_id=channel['channel_id'],
                api_key=channel['api_key'],
                results=5000
            )
            
            if success:
                generator.generate_all()
                results.append({
                    'channel': channel['name'],
                    'status': 'success',
                    'records': len(generator.df),
                    'output': output_dir
                })
                print(f"  ✓ Generated website: {output_dir}/index.html")
            else:
                results.append({
                    'channel': channel['name'],
                    'status': 'failed',
                    'error': 'Failed to fetch data'
                })
                print(f"  ✗ Failed to fetch data")
        
        return results
    
    def process_combined(self):
        """Fetch all channels and create combined analysis"""
        print("\n" + "="*80)
        print("PROCESSING COMBINED ANALYSIS")
        print("="*80)
        
        all_data = []
        
        for idx, channel in enumerate(self.channels, 1):
            if not channel.get('enabled', True):
                continue
            
            print(f"\n[{idx}] Fetching: {channel['name']}")
            
            generator = WiSUNWebsiteGenerator()
            success = generator.fetch_data_from_thingspeak(
                channel_id=channel['channel_id'],
                api_key=channel['api_key'],
                results=5000
            )
            
            if success:
                generator.df['location'] = channel['name']
                all_data.append(generator.df)
                print(f"  ✓ Fetched {len(generator.df)} records")
            else:
                print(f"  ✗ Failed to fetch")
        
        if not all_data:
            print("\n✗ No data fetched from any channel")
            return False
        
        # Combine all data
        print(f"\n📊 Combining data from {len(all_data)} channels...")
        combined_generator = WiSUNWebsiteGenerator(output_dir='batch_output/combined_analysis')
        combined_generator.df = pd.concat(all_data, ignore_index=True)
        
        print(f"  Total records: {len(combined_generator.df)}")
        print(f"  Date range: {combined_generator.df['timestamp'].min()} to {combined_generator.df['timestamp'].max()}")
        
        combined_generator.generate_all()
        print(f"\n✓ Combined analysis generated: batch_output/combined_analysis/index.html")
        
        return True
    
    def generate_index_page(self, results):
        """Generate master index page linking to all reports"""
        print("\n📝 Generating master index page...")
        
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wi-SUN Batch Analysis - Index</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        header {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            text-align: center;
            margin-bottom: 30px;
        }
        h1 { color: #667eea; font-size: 3em; margin-bottom: 10px; }
        .subtitle { color: #666; font-size: 1.2em; }
        .reports-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
        }
        .report-card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .report-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .report-title {
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 15px;
            font-weight: bold;
        }
        .report-info {
            color: #666;
            margin: 10px 0;
            font-size: 1em;
        }
        .report-link {
            display: inline-block;
            margin-top: 20px;
            padding: 12px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s;
        }
        .report-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        .status-success { color: #4caf50; font-weight: bold; }
        .status-failed { color: #f44336; font-weight: bold; }
        .combined-section {
            background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
            color: white;
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            text-align: center;
        }
        .combined-section h2 { margin-bottom: 15px; font-size: 2em; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌐 Wi-SUN Batch Analysis</h1>
            <p class="subtitle">Multi-Channel Network Analysis Dashboard</p>
        </header>
"""
        
        # Combined analysis section
        if Path('batch_output/combined_analysis/index.html').exists():
            html += """
        <div class="combined-section">
            <h2>📊 Combined Analysis</h2>
            <p style="margin-bottom: 20px;">Comprehensive analysis across all channels</p>
            <a href="combined_analysis/index.html" class="report-link" style="background: white; color: #4caf50;">
                View Combined Report →
            </a>
        </div>
"""
        
        html += """
        <h2 style="color: white; margin-bottom: 20px; font-size: 2em; text-align: center;">
            Individual Channel Reports
        </h2>
        <div class="reports-grid">
"""
        
        for result in results:
            status_class = "status-success" if result['status'] == 'success' else "status-failed"
            status_text = "✓ Success" if result['status'] == 'success' else "✗ Failed"
            
            if result['status'] == 'success':
                html += f"""
            <div class="report-card">
                <div class="report-title">{result['channel']}</div>
                <p class="report-info">Status: <span class="{status_class}">{status_text}</span></p>
                <p class="report-info">Records: {result['records']:,}</p>
                <a href="{result['channel']}/index.html" class="report-link">View Dashboard →</a>
            </div>
"""
            else:
                html += f"""
            <div class="report-card" style="opacity: 0.6;">
                <div class="report-title">{result['channel']}</div>
                <p class="report-info">Status: <span class="{status_class}">{status_text}</span></p>
                <p class="report-info" style="color: #f44336;">{result.get('error', 'Unknown error')}</p>
            </div>
"""
        
        html += """
        </div>
    </div>
</body>
</html>
"""
        
        output_file = Path('batch_output/index.html')
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(html)
        
        print(f"✓ Master index generated: {output_file}")
    
    def run(self):
        """Run the batch processor"""
        if not self.load_config():
            return
        
        enabled_channels = [c for c in self.channels if c.get('enabled', True)]
        
        if not enabled_channels:
            print("✗ No enabled channels found in config")
            return
        
        print(f"\n🚀 Starting batch processing for {len(enabled_channels)} channels...")
        
        # Process individual channels
        results = self.process_individual_channels()
        
        # Process combined analysis
        self.process_combined()
        
        # Generate master index
        self.generate_index_page(results)
        
        print("\n" + "="*80)
        print("✅ BATCH PROCESSING COMPLETE!")
        print("="*80)
        print(f"\n📂 Output directory: batch_output/")
        print(f"🌐 Master index: batch_output/index.html")
        print(f"\n✓ Successfully processed: {sum(1 for r in results if r['status'] == 'success')}/{len(results)} channels")


def main():
    """Main execution"""
    print("""
╔════════════════════════════════════════════════════════════════╗
║         ThingSpeak Batch Processor for Wi-SUN Analysis        ║
║                                                                 ║
║  Process multiple ThingSpeak channels simultaneously           ║
╚════════════════════════════════════════════════════════════════╝
    """)
    
    processor = ThingSpeakBatchProcessor()
    processor.run()


if __name__ == "__main__":
    main()
