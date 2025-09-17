#!/usr/bin/env python3
"""
Setup script for Plant Disease Detection Backend
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🌱 Plant Disease Detection Backend Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python {sys.version.split()[0]} detected")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Check if Kaggle credentials exist
    kaggle_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_path):
        print("\n⚠️  Kaggle API credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("4. Run: chmod 600 ~/.kaggle/kaggle.json")
        
        response = input("\nDo you want to continue without downloading the dataset? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print("✓ Kaggle credentials found")
        
        # Download dataset
        if run_command("python download_dataset.py", "Downloading PlantVillage dataset"):
            print("✓ Dataset downloaded successfully")
            
            # Train model
            train_response = input("\nDo you want to train the model now? This may take 30-60 minutes (y/n): ")
            if train_response.lower() == 'y':
                if run_command("python train_model.py", "Training plant disease model"):
                    print("✓ Model training completed")
                else:
                    print("❌ Model training failed")
        else:
            print("❌ Dataset download failed")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. If you haven't trained the model yet, run: python train_model.py")
    print("2. Start the API server: uvicorn api:app --reload")
    print("3. The API will be available at http://localhost:8000")
    print("4. API documentation at http://localhost:8000/docs")

if __name__ == "__main__":
    main()