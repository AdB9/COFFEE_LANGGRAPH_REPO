#!/usr/bin/env python3
"""Test Sentinel Hub authentication specifically."""

import os
from dotenv import load_dotenv
from sentinelhub.config import SHConfig
from sentinelhub.api.process import SentinelHubRequest
from sentinelhub.data_collections import DataCollection
from sentinelhub.constants import MimeType, CRS
from sentinelhub.geometry import BBox
from sentinelhub.geo_utils import bbox_to_dimensions

# Load environment variables
load_dotenv()

print("=== Sentinel Hub Authentication Test ===")

# Get credentials
sh_client_id = os.getenv('SH_CLIENT_ID')
sh_client_secret = os.getenv('SH_CLIENT_SECRET')

print(f"Client ID: {sh_client_id[:10] if sh_client_id else 'None'}... (length: {len(sh_client_id) if sh_client_id else 0})")
print(f"Client Secret: {sh_client_secret[:10] if sh_client_secret else 'None'}... (length: {len(sh_client_secret) if sh_client_secret else 0})")

# Create config
config = SHConfig()
if sh_client_id:
    config.sh_client_id = sh_client_id
if sh_client_secret:
    config.sh_client_secret = sh_client_secret

print(f"Config instance URL: {config.sh_base_url}")
print(f"Config auth URL: {config.sh_auth_base_url}")

# Test a simple data collection request
try:
    print("\n=== Testing Data Collections ===")
    available_collections = [
        DataCollection.SENTINEL2_L1C,
        DataCollection.SENTINEL2_L2A,
    ]
    
    for collection in available_collections:
        print(f"Collection: {collection}")
        print(f"  Service URL: {collection.service_url}")
        print(f"  Bands: {getattr(collection, 'bands', 'N/A')}")

except Exception as e:
    print(f"❌ Error testing collections: {e}")

# Test a minimal request
try:
    print("\n=== Testing Minimal Request ===")
    
    # Simple bounding box (small area in Brazil)
    bbox = BBox((-44.0, -20.0, -43.9, -19.9), crs=CRS.WGS84)
    size = (100, 100)  # Small size for testing
    
    # Simple evalscript
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3 }
        };
    }
    function evaluatePixel(sample) {
        return [sample.B04, sample.B03, sample.B02];
    }
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=('2024-08-01', '2024-08-02'),
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=size,
        config=config,
    )
    
    print("✅ Request created successfully")
    print(f"Request URL pattern: {request}")
    
    # Try to get data
    print("Attempting to fetch data...")
    data = request.get_data()
    print(f"✅ Data retrieved successfully! Got {len(data)} items")
    if data and len(data) > 0:
        print(f"First item shape: {data[0].shape if hasattr(data[0], 'shape') else 'No shape attr'}")
    
except Exception as e:
    print(f"❌ Error with request: {e}")
    print(f"Error type: {type(e)}")
    
    # Check if it's an authentication error
    if "invalid_client" in str(e).lower():
        print("🔍 This appears to be an authentication issue.")
        print("Possible causes:")
        print("1. Client ID or secret is incorrect")
        print("2. Credentials have expired")
        print("3. Account doesn't have proper permissions")
        print("4. Sentinel Hub service is down")
        
print("\n=== Test Complete ===")
