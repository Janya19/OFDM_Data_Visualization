#!/bin/bash
# Quick commands for Wi-SUN OFDM Analysis

echo "═══════════════════════════════════════════════════════"
echo "  Wi-SUN OFDM Network Analysis - Quick Commands"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "📂 Current Directory: main_ofdm/"
echo ""
echo "Available Commands:"
echo ""
echo "  1. 🌐 Open Website"
echo "     xdg-open wisun_website_output/index.html"
echo ""
echo "  2. 🖥️  Start Local Server"
echo "     cd wisun_website_output && python -m http.server 8000"
echo "     Then open: http://localhost:8000"
echo ""
echo "  3. 🔄 Regenerate Website"
echo "     python generate_website.py"
echo ""
echo "  4. 📊 View Statistics"
echo "     cat wisun_website_output/summary_statistics.csv"
echo ""
echo "  5. 📁 List Data Files"
echo "     ls -lh Locations/"
echo ""
echo "  6. 📖 Read Documentation"
echo "     cat README.md"
echo ""
echo "═══════════════════════════════════════════════════════"
echo ""

# If argument provided, execute the command
case "$1" in
    open|view|1)
        xdg-open wisun_website_output/index.html
        ;;
    server|serve|2)
        cd wisun_website_output && python -m http.server 8000
        ;;
    generate|regen|3)
        python generate_website.py
        ;;
    stats|4)
        cat wisun_website_output/summary_statistics.csv
        ;;
    list|ls|5)
        ls -lh Locations/
        ;;
    help|readme|6)
        cat README.md
        ;;
    *)
        echo "Usage: ./COMMANDS.sh [open|server|generate|stats|list|help]"
        echo "Or run without arguments to see the menu"
        ;;
esac
