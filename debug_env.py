#!/usr/bin/env python3
"""Debug script to check environment variable loading."""

import os
from dotenv import load_dotenv

print("=== Environment Debug ===")

print(f"Current working directory: {os.getcwd()}")

# Check if .env file exists
env_file = ".env"
if os.path.exists(env_file):
    print(f"✅ {env_file} exists")
    with open(env_file, 'r') as f:
        content = f.read()
        lines = content.strip().split('\n')
        print(f"   File has {len(lines)} lines")
        for i, line in enumerate(lines, 1):
            if line.strip() and not line.strip().startswith('#'):
                key = line.split('=')[0] if '=' in line else line
                print(f"   Line {i}: {key}=...")
else:
    print(f"❌ {env_file} not found")

print("\n=== Before load_dotenv() ===")
print(f"SH_CLIENT_ID: {os.getenv('SH_CLIENT_ID')}")
print(f"SH_CLIENT_SECRET: {os.getenv('SH_CLIENT_SECRET')}")

print("\n=== Loading .env ===")
load_result = load_dotenv()
print(f"load_dotenv() returned: {load_result}")

print("\n=== After load_dotenv() ===")
sh_client_id = os.getenv('SH_CLIENT_ID')
sh_client_secret = os.getenv('SH_CLIENT_SECRET')

print(f"SH_CLIENT_ID: {sh_client_id[:8] + '...' if sh_client_id else 'None'}")
print(f"SH_CLIENT_SECRET: {sh_client_secret[:8] + '...' if sh_client_secret else 'None'}")

if sh_client_id and sh_client_secret:
    print("✅ Credentials loaded successfully!")
else:
    print("❌ Credentials not loaded properly")

print("\n=== Sentinel Hub Config Test ===")
try:
    from sentinelhub.config import SHConfig
    config = SHConfig()
    
    if sh_client_id and sh_client_secret:
        config.sh_client_id = sh_client_id
        config.sh_client_secret = sh_client_secret
        print("✅ Config set successfully")
        print(f"Config client ID: {config.sh_client_id[:8] + '...' if config.sh_client_id else 'None'}")
    else:
        print("❌ Cannot set config - missing credentials")
        
except Exception as e:
    print(f"❌ Error with SentinelHub config: {e}")
