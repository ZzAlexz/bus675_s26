"""
Congo Returns — Preprocessor / Folder Watcher

Monitors a folder for incoming product photos, extracts metadata from
filenames, and sends them to the inference API for classification.

This simulates the warehouse datacenter component that handles incoming
product photos from the automated unboxing stations.

Expected filename format: CUST{customer_id}_PROD{product_id}_{description}.jpg
Example: CUST12345_PROD67890_laptop_charger.jpg
"""

import os
import re
import sys
import time
import shutil
import requests
from pathlib import Path
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

# The folder to watch for incoming images
# In Docker, this should be mounted from the host
WATCH_FOLDER = Path("/incoming")

# Folder for successfully processed images
PROCESSED_FOLDER = Path("/incoming/processed")

# Get the API URL from environment variable
# This allows configuration at container runtime
API_URL = os.environ.get("API_URL", "http://localhost:8000")

# How often to check for new files (seconds)
POLL_INTERVAL = 2

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}


# ============================================================================
# Filename Parsing
# ============================================================================

def parse_filename(filename: str) -> dict:
    """
    Extract customer_id and product_id from Congo's filename format.
    
    Expected format: CUST{customer_id}_PROD{product_id}_{description}.ext
    Example: CUST12345_PROD67890_wireless_mouse.jpg
    
    Args:
        filename: The filename to parse
        
    Returns:
        Dictionary with customer_id, product_id, and description
        Returns None values if parsing fails
    """
    pattern = r'^CUST(\d+)_PROD(\d+)_(.+)\.\w+$'
    match = re.match(pattern, filename)
    
    if match:
        return {
            "customer_id": match.group(1),
            "product_id": match.group(2),
            "description": match.group(3)
        }
    else:
        # Filename doesn't match expected format
        return {
            "customer_id": None,
            "product_id": None,
            "description": None
        }


# ============================================================================
# API Communication
# ============================================================================

def send_to_api(image_path: Path, metadata: dict) -> dict:
    """
    Send an image to the inference API for classification.
    
    Args:
        image_path: Path to the image file
        metadata: Dictionary with customer_id and product_id
        
    Returns:
        API response as dictionary, or None if request failed
    """
    url = f"{API_URL}/predict"
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (image_path.name, f, 'image/jpeg')}
            params = {
                'customer_id': metadata.get('customer_id'),
                'product_id': metadata.get('product_id')
            }
            
            response = requests.post(url, files=files, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"  ❌ API error {response.status_code}: {response.text}")
                return None
                
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Cannot connect to API at {url}")
        return None
    except requests.exceptions.Timeout:
        print(f"  ❌ API request timed out")
        return None
    except Exception as e:
        print(f"  ❌ Error sending to API: {e}")
        return None


# ============================================================================
# Image Processing
# ============================================================================

def process_image(image_path: Path) -> bool:
    """
    Process a single image: parse metadata, send to API, move to processed.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        True if successful, False otherwise
    """
    filename = image_path.name
    print(f"  Processing: {filename}")
    
    # Parse metadata from filename
    metadata = parse_filename(filename)
    
    if metadata["customer_id"]:
        print(f"    Customer: {metadata['customer_id']}, Product: {metadata['product_id']}")
    else:
        print(f"    Warning: Filename doesn't match expected format")
    
    # Send to inference API
    result = send_to_api(image_path, metadata)
    
    if result:
        print(f"    ✅ Classified as: {result['top_prediction']} ({result['confidence']}%)")
        
        # Move to processed folder
        PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)
        dest_path = PROCESSED_FOLDER / filename
        shutil.move(str(image_path), str(dest_path))
        
        return True
    else:
        print(f"    ⚠️  Failed to classify, will retry later")
        return False


def get_pending_images() -> list:
    """
    Get list of image files waiting to be processed.
    
    Returns:
        List of Path objects for pending images
    """
    if not WATCH_FOLDER.exists():
        return []
    
    pending = []
    for file_path in WATCH_FOLDER.iterdir():
        # Skip directories and processed folder
        if file_path.is_dir():
            continue
            
        # Check if it's an image
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            pending.append(file_path)
    
    return sorted(pending)  # Process in alphabetical order


# ============================================================================
# Main Loop
# ============================================================================

def main():
    """
    Main watcher loop. Continuously monitors for new images.
    """
    print("=" * 60)
    print("Congo Returns - Preprocessor / Folder Watcher")
    print("=" * 60)
    print(f"Watching folder: {WATCH_FOLDER}")
    print(f"API URL: {API_URL}")
    print(f"Poll interval: {POLL_INTERVAL} seconds")
    print("=" * 60)
    
    # Ensure watch folder exists
    WATCH_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Check API connectivity at startup
    print("\nChecking API connectivity...")
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✅ API is reachable at {API_URL}")
        else:
            print(f"⚠️  API returned status {response.status_code}")
    except Exception as e:
        print(f"⚠️  Cannot reach API: {e}")
        print("   Will retry when processing images...")
    
    print("\nWaiting for images...")
    print("-" * 60)
    
    # Main polling loop
    while True:
        try:
            pending = get_pending_images()
            
            if pending:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(pending)} image(s) to process")
                
                for image_path in pending:
                    process_image(image_path)
                    
                print("-" * 60)
            
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            print("\nShutting down...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
