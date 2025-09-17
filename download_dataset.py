import os
import kaggle
import zipfile
import shutil

def download_plantvillage_dataset():
    """Download and extract PlantVillage dataset from Kaggle"""
    
    print("Downloading PlantVillage dataset from Kaggle...")
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    try:
        # Download dataset using Kaggle API
        kaggle.api.dataset_download_files(
            'emmarex/plantdisease', 
            path='data/', 
            unzip=True
        )
        print("Dataset downloaded and extracted successfully!")
        
        # Check if extraction was successful
        if os.path.exists("data/PlantVillage"):
            print("PlantVillage dataset found!")
            return True
        else:
            print("Dataset structure might be different. Checking available folders...")
            for item in os.listdir("data"):
                print(f"Found: {item}")
            return False
            
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have:")
        print("1. Kaggle API credentials configured (~/.kaggle/kaggle.json)")
        print("2. Internet connection")
        print("3. Accepted the dataset terms on Kaggle website")
        return False

if __name__ == "__main__":
    success = download_plantvillage_dataset()
    if success:
        print("Ready to train the model!")
    else:
        print("Please resolve the issues above before training.")