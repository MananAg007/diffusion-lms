#!/usr/bin/env python3
"""Pre-download LM1B dataset using wget/curl with proper resume support."""

import os
import subprocess
import sys
import tarfile
import shutil

# Cache directory from config
CACHE_DIR = "/data2/david3684/.cache/huggingface/datasets"

# Original download URLs (both HTTP and HTTPS - datasets uses HTTP)
DOWNLOAD_URL_HTTPS = "https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
DOWNLOAD_URL_HTTP = "http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
DOWNLOAD_URL = DOWNLOAD_URL_HTTPS  # For downloading, use HTTPS
TAR_FILENAME = "1-billion-word-language-modeling-benchmark-r13output.tar.gz"
EXPECTED_SIZE = 1792209805  # bytes (~1.67 GB)


def check_command(cmd):
    """Check if a command exists."""
    try:
        subprocess.run(['which', cmd], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def download_with_wget(url, output_path):
    """Download using wget with resume support."""
    print(f"Using wget to download...")
    cmd = [
        'wget',
        '-c',  # Continue/resume download
        '--timeout=60',
        '--tries=0',  # Infinite retries
        '--continue',
        '--progress=bar:force',
        '--no-check-certificate',  # In case of SSL issues
        '-O', output_path,
        url
    ]
    return subprocess.run(cmd, check=False)


def download_with_curl(url, output_path):
    """Download using curl with resume support."""
    print(f"Using curl to download...")
    cmd = [
        'curl',
        '-L',  # Follow redirects (HTTP -> HTTPS)
        '-C', '-',  # Resume from previous download
        '--connect-timeout', '60',
        '--max-time', '0',  # No timeout
        '--retry', '999',  # Retry many times
        '--retry-delay', '5',
        '--progress-bar',
        '-o', output_path,
        url
    ]
    return subprocess.run(cmd, check=False)


def download_file(url, output_path):
    """Download file with resume support using wget or curl."""
    # Check if file already exists and is complete
    if os.path.exists(output_path):
        size = os.path.getsize(output_path)
        if size == EXPECTED_SIZE:
            print(f"✅ File already exists and is complete: {output_path}")
            return True
        elif size > 0:
            print(f"📥 Resuming download from {size:,} bytes ({size/1024/1024:.1f} MB)...")
    
    # Try wget first, then curl
    if check_command('wget'):
        result = download_with_wget(url, output_path)
        if result.returncode == 0:
            return True
        print("⚠️  wget failed, trying curl...")
    
    if check_command('curl'):
        result = download_with_curl(url, output_path)
        if result.returncode == 0:
            return True
        print("⚠️  curl also failed...")
    
    return False


def main():
    print(f"{'='*60}")
    print("LM1B Dataset Downloader (Direct Download with Resume)")
    print(f"{'='*60}")
    print(f"URL: {DOWNLOAD_URL}")
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Expected size: {EXPECTED_SIZE:,} bytes (~{EXPECTED_SIZE/1024/1024/1024:.2f} GB)")
    print(f"\n⚠️  This will download the raw tar.gz file.")
    print(f"   If interrupted, rerun this script to resume.")
    
    # Create cache directory
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    # Download to a temporary location first
    temp_dir = os.path.join(CACHE_DIR, 'lm1b_raw')
    os.makedirs(temp_dir, exist_ok=True)
    tar_path = os.path.join(temp_dir, TAR_FILENAME)
    
    print(f"\n{'='*60}")
    print("Step 1: Downloading tar.gz file...")
    print(f"{'='*60}")
    
    if not download_file(DOWNLOAD_URL, tar_path):
        print("\n❌ Download failed!")
        print("   You can manually download using:")
        print(f"   wget -c {DOWNLOAD_URL} -O {tar_path}")
        print(f"   or")
        print(f"   curl -C - {DOWNLOAD_URL} -o {tar_path}")
        sys.exit(1)
    
    # Verify file size
    actual_size = os.path.getsize(tar_path)
    if actual_size != EXPECTED_SIZE:
        print(f"\n⚠️  Warning: File size mismatch!")
        print(f"   Expected: {EXPECTED_SIZE:,} bytes")
        print(f"   Actual: {actual_size:,} bytes")
        print(f"   File may be incomplete. Please check.")
    
    print(f"\n✅ Download completed: {tar_path}")
    print(f"   Size: {actual_size:,} bytes ({actual_size/1024/1024/1024:.2f} GB)")
    
    print(f"\n{'='*60}")
    print("Step 2: Copying file to datasets library cache...")
    print(f"{'='*60}")
    
    # Get the exact filename that datasets library uses for BOTH HTTP and HTTPS
    # datasets library uses HTTP URL, but we downloaded with HTTPS
    try:
        from datasets.utils.file_utils import hash_url_to_filename
        datasets_filename_http = hash_url_to_filename(DOWNLOAD_URL_HTTP)
        datasets_filename_https = hash_url_to_filename(DOWNLOAD_URL_HTTPS)
    except ImportError:
        # Fallback: use SHA256 hash
        import hashlib
        datasets_filename_http = hashlib.sha256(DOWNLOAD_URL_HTTP.encode()).hexdigest()
        datasets_filename_https = hashlib.sha256(DOWNLOAD_URL_HTTPS.encode()).hexdigest()
    
    downloads_dir = os.path.join(CACHE_DIR, 'downloads')
    os.makedirs(downloads_dir, exist_ok=True)
    
    # Remove any incomplete downloads
    import glob
    incomplete_files = glob.glob(os.path.join(downloads_dir, '*.incomplete'))
    lock_files = glob.glob(os.path.join(downloads_dir, '*.lock'))
    for f in incomplete_files + lock_files:
        try:
            os.remove(f)
            print(f"   Removed: {os.path.basename(f)}")
        except:
            pass
    
    # Copy file to BOTH HTTP and HTTPS cache locations
    # (datasets library uses HTTP, but we downloaded with HTTPS)
    for url_type, datasets_filename in [("HTTP", datasets_filename_http), ("HTTPS", datasets_filename_https)]:
        target_path = os.path.join(downloads_dir, datasets_filename)
        
        if os.path.exists(target_path):
            existing_size = os.path.getsize(target_path)
            if existing_size == EXPECTED_SIZE:
                print(f"✅ {url_type} file already exists: {os.path.basename(target_path)}")
                continue
            else:
                print(f"⚠️  Removing incomplete {url_type} file: {os.path.basename(target_path)}")
                os.remove(target_path)
        
        shutil.copy2(tar_path, target_path)
        copied_size = os.path.getsize(target_path)
        print(f"✅ Copied to {url_type} cache: {os.path.basename(target_path)}")
        print(f"   Size: {copied_size:,} bytes ({copied_size/1024/1024/1024:.2f} GB)")
    
    print(f"\n✅ Setup complete!")
    print(f"   Raw file: {tar_path}")
    print(f"   Datasets cache: {target_path}")
    print(f"   You can now use datasets.load_dataset('lm1b', ...)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Download interrupted by user.")
        print("   Progress has been saved. Rerun this script to resume.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        sys.exit(1)

