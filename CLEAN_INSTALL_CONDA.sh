#!/bin/bash
# Clean install of Miniconda3 in /project/flame/mananaga
# This script will remove the old conda and install fresh

set -e  # Exit on error

echo "=========================================="
echo "Clean Conda Installation"
echo "=========================================="

# Step 1: Deactivate conda if active
echo "Step 1: Deactivating conda..."
conda deactivate 2>/dev/null || true
# Deactivate multiple times to ensure we're out
conda deactivate 2>/dev/null || true

# Step 2: Remove conda from PATH for this session
echo "Step 2: Removing conda from PATH..."
export PATH=$(echo $PATH | tr ':' '\n' | grep -v miniconda3 | tr '\n' ':' | sed 's/:$//')

# Step 3: Remove conda initialization from .bashrc
echo "Step 3: Removing conda initialization from ~/.bashrc..."
if [ -f ~/.bashrc ]; then
    # Create backup
    cp ~/.bashrc ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)
    echo "Created backup: ~/.bashrc.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Remove conda initialization block
    sed -i '/# >>> conda initialize >>>/,/# <<< conda initialize <<</d' ~/.bashrc
    echo "Removed conda initialization block from ~/.bashrc"
fi

# Step 4: Remove or rename old miniconda3
echo "Step 4: Removing old miniconda3..."
if [ -d ~/miniconda3 ]; then
    read -p "Remove ~/miniconda3? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ~/miniconda3
        echo "Removed ~/miniconda3"
    else
        mv ~/miniconda3 ~/miniconda3.old.$(date +%Y%m%d_%H%M%S)
        echo "Renamed ~/miniconda3 to ~/miniconda3.old.$(date +%Y%m%d_%H%M%S)"
    fi
fi

# Remove .conda if it exists
if [ -d ~/.conda ]; then
    read -p "Remove ~/.conda? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ~/.conda
        echo "Removed ~/.conda"
    else
        mv ~/.conda ~/.conda.old.$(date +%Y%m%d_%H%M%S)
        echo "Renamed ~/.conda to ~/.conda.old.$(date +%Y%m%d_%H%M%S)"
    fi
fi

# Step 5: Create target directory
echo "Step 5: Creating target directory..."
mkdir -p /project/flame/mananaga
cd /project/flame/mananaga

# Step 6: Download Miniconda3 if not already present
echo "Step 6: Downloading Miniconda3..."
if [ ! -f Miniconda3-latest-Linux-x86_64.sh ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
else
    echo "Miniconda3 installer already exists, using existing file"
fi

# Step 7: Install Miniconda3 in new location
echo "Step 7: Installing Miniconda3 in /project/flame/mananaga/miniconda3..."
bash Miniconda3-latest-Linux-x86_64.sh -b -p /project/flame/mananaga/miniconda3

# Step 8: Initialize conda
echo "Step 8: Initializing conda..."
/project/flame/mananaga/miniconda3/bin/conda init bash

# Step 9: Source the updated .bashrc
echo "Step 9: Sourcing updated .bashrc..."
source ~/.bashrc

# Step 10: Configure conda
echo "Step 10: Configuring conda..."
conda config --set auto_activate_base false
conda config --add envs_dirs /project/flame/mananaga/conda/envs
conda config --add envs_dirs /project/flame/mananaga/.conda/envs

# Step 11: Update LOCAL_DIR in .bashrc
echo "Step 11: Setting LOCAL_DIR..."
if ! grep -q "export LOCAL_DIR=" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# TESS-Diffusion LOCAL_DIR" >> ~/.bashrc
    echo "export LOCAL_DIR=/project/flame/mananaga" >> ~/.bashrc
    echo "Added LOCAL_DIR to ~/.bashrc"
else
    sed -i 's|export LOCAL_DIR=.*|export LOCAL_DIR=/project/flame/mananaga|' ~/.bashrc
    echo "Updated LOCAL_DIR in ~/.bashrc"
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Open a new terminal or run: source ~/.bashrc"
echo "2. Verify conda: conda --version"
echo "3. Create environments in the new location:"
echo "   export LOCAL_DIR=/project/flame/mananaga"
echo "   conda env create -f environment.yaml --prefix \${LOCAL_DIR}/conda/envs/env_name"
echo ""
echo "To verify installation:"
echo "  conda info"
echo "  conda config --show envs_dirs"
echo ""

