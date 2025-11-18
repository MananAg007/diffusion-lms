# Guide to Move Conda from /home/mananaga to /project/flame/mananaga

## Overview

You have two main options:
- **Option A**: Move just environments (keeps base conda in `/home/mananaga/miniconda3`)
- **Option B**: Move entire conda installation (moves everything including base)

## Current Setup

- Base conda: `/home/mananaga/miniconda3`
- Environments: `/home/mananaga/miniconda3/envs` (e.g., `duo`)
- Conda config envs_dirs: `/home/mananaga/miniconda3/envs` and `/home/mananaga/.conda/envs`

---

## Option A: Move Only Environments (Recommended)

This keeps your base conda installation but moves all environments to the new location.

### Step 1: Create new directory structure
```bash
mkdir -p /project/flame/mananaga/conda/envs
mkdir -p /project/flame/mananaga/.conda/envs
```

### Step 2: Move existing environments
```bash
# Move environments from miniconda3/envs
mv /home/mananaga/miniconda3/envs/* /project/flame/mananaga/conda/envs/ 2>/dev/null || true

# Move environments from .conda/envs if any exist
if [ -d "/home/mananaga/.conda/envs" ] && [ "$(ls -A /home/mananaga/.conda/envs 2>/dev/null)" ]; then
    mv /home/mananaga/.conda/envs/* /project/flame/mananaga/.conda/envs/
fi
```

### Step 3: Update conda configuration
```bash
# Remove old envs_dirs
conda config --remove envs_dirs /home/mananaga/miniconda3/envs
conda config --remove envs_dirs /home/mananaga/.conda/envs

# Add new envs_dirs (in order of preference)
conda config --add envs_dirs /project/flame/mananaga/conda/envs
conda config --add envs_dirs /project/flame/mananaga/.conda/envs

# Verify
conda config --show envs_dirs
```

### Step 4: Deactivate and reactivate environments
```bash
conda deactivate
conda activate duo  # Should now find it in the new location
```

### Step 5: Verify environment paths
```bash
conda info --envs
# Should show environments pointing to /project/flame/mananaga/conda/envs/
```

---

## Option B: Move Entire Conda Installation

This moves everything including the base conda installation.

### Step 1: Deactivate all conda environments
```bash
conda deactivate  # Repeat until base is active or no conda is active
```

### Step 2: Create new directory and move miniconda3
```bash
# Create target directory
mkdir -p /project/flame/mananaga

# Move entire miniconda3
mv /home/mananaga/miniconda3 /project/flame/mananaga/

# Move .conda if it exists
if [ -d "/home/mananaga/.conda" ]; then
    mv /home/mananaga/.conda /project/flame/mananaga/
fi
```

### Step 3: Update shell configuration

Edit your `~/.bashrc` or `~/.bash_profile`:

**Find and update the conda initialization block:**
```bash
# Find this block (usually at the end):
# >>> conda initialize >>>
# ... conda initialization code ...
# <<< conda initialize <<<

# Update the path from:
# /home/mananaga/miniconda3

# To:
# /project/flame/mananaga/miniconda3
```

Or replace the entire conda init block:
```bash
# Remove old conda init (if present)
# Then re-initialize:
/project/flame/mananaga/miniconda3/bin/conda init bash
```

### Step 4: Update conda configuration paths
```bash
# Source the updated shell config or start new shell
source ~/.bashrc

# Update envs_dirs
conda config --remove envs_dirs /home/mananaga/miniconda3/envs
conda config --add envs_dirs /project/flame/mananaga/miniconda3/envs
conda config --remove envs_dirs /home/mananaga/.conda/envs
conda config --add envs_dirs /project/flame/mananaga/.conda/envs

# Verify
conda config --show envs_dirs
conda info --envs
```

---

## After Moving: Update Project Configurations

### Step 1: Update LOCAL_DIR in your shell config

Add to your `~/.bashrc` or `~/.bash_profile`:
```bash
export LOCAL_DIR=/project/flame/mananaga
```

### Step 2: Update TESS-Diffusion paths

If you've already created environments using the old path, you'll need to:
1. Either recreate them with the new LOCAL_DIR, or
2. Move them manually

For new environments in TESS-Diffusion:
```bash
export LOCAL_DIR=/project/flame/mananaga
cd /project/flame/mananaga/diffusion-lms/tess-diffusion  # Update if you move this too
conda env create -f environment.yaml --prefix ${LOCAL_DIR}/conda/envs/sdlm
```

### Step 3: Verify everything works
```bash
# Check conda is working
conda --version

# Check environments
conda env list

# Activate an environment
conda activate duo  # or your environment name

# Verify LOCAL_DIR
echo $LOCAL_DIR  # Should show /project/flame/mananaga
```

---

## Quick Commands Summary (Option A - Recommended)

```bash
# 1. Create directories
mkdir -p /project/flame/mananaga/conda/envs /project/flame/mananaga/.conda/envs

# 2. Move environments
mv /home/mananaga/miniconda3/envs/* /project/flame/mananaga/conda/envs/ 2>/dev/null || true
[ -d "/home/mananaga/.conda/envs" ] && mv /home/mananaga/.conda/envs/* /project/flame/mananaga/.conda/envs/ 2>/dev/null || true

# 3. Update conda config
conda config --remove envs_dirs /home/mananaga/miniconda3/envs
conda config --remove envs_dirs /home/mananaga/.conda/envs
conda config --add envs_dirs /project/flame/mananaga/conda/envs
conda config --add envs_dirs /project/flame/mananaga/.conda/envs

# 4. Update shell config
echo 'export LOCAL_DIR=/project/flame/mananaga' >> ~/.bashrc

# 5. Reload shell
source ~/.bashrc

# 6. Verify
conda info --envs
conda activate duo
```

---

## Troubleshooting

### If environments don't activate:
```bash
# Check if they exist in new location
ls -la /project/flame/mananaga/conda/envs/

# Check conda config
conda config --show envs_dirs

# Recreate environment if needed
conda env create -f environment.yaml --prefix /project/flame/mananaga/conda/envs/environment_name
```

### If conda command not found after Option B:
```bash
# Add to PATH manually
export PATH="/project/flame/mananaga/miniconda3/bin:$PATH"

# Then reinitialize
/project/flame/mananaga/miniconda3/bin/conda init bash
source ~/.bashrc
```

### Clean up old directories (after verifying everything works):
```bash
# Only do this after confirming everything works!
# Option A cleanup:
rmdir /home/mananaga/miniconda3/envs 2>/dev/null || true
rmdir /home/mananaga/.conda/envs 2>/dev/null || true

# Option B cleanup: Don't remove miniconda3 yet - keep as backup for a while
```

