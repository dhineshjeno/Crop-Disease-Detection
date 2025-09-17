from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
from io import BytesIO

app = FastAPI(title="Plant Disease Detection API", version="1.0.0")

# CORS configuration for React frontend
origins = [
    "http://127.0.0.1:5174",
    "http://localhost:5174",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and labels
model = None
class_labels = []
IMG_SIZE = 64

def load_model_and_labels():
    """Load the trained model and class labels"""
    global model, class_labels
    
    try:
        # Load model
        if os.path.exists("plant_disease_model.h5"):
            model = tf.keras.models.load_model("plant_disease_model.h5")
            print("Model loaded successfully!")
        else:
            print("Model file not found. Please train the model first.")
            return False
        
        # Load class labels
        if os.path.exists("class_labels.json"):
            with open("class_labels.json", "r") as f:
                class_labels = json.load(f)
            print(f"Loaded {len(class_labels)} class labels")
        else:
            print("Class labels file not found.")
            return False
            
        return True
        
    except Exception as e:
        print(f"Error loading model or labels: {e}")
        return False

def preprocess_image(image: Image.Image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error preprocessing image: {str(e)}")

def is_plant_image(predictions, confidence_threshold=0.1):
    """Check if the image is likely a plant/crop image"""
    max_confidence = np.max(predictions)
    return max_confidence > confidence_threshold

def format_disease_name(class_name):
    """Format the class name to be more readable"""
    # Remove common prefixes and format
    formatted = class_name.replace("___", " - ").replace("__", " ").replace("_", " ")
    
    # Capitalize words
    words = formatted.split()
    formatted_words = []
    for word in words:
        if word.lower() in ['healthy', 'disease', 'spot', 'blight', 'rust', 'mildew']:
            formatted_words.append(word.capitalize())
        else:
            formatted_words.append(word.title())
    
    return " ".join(formatted_words)

@app.on_event("startup")
async def startup_event():
    """Load model and labels on startup"""
    success = load_model_and_labels()
    if not success:
        print("Warning: Model or labels could not be loaded. API will not work properly.")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Plant Disease Detection API",
        "status": "running",
        "model_loaded": model is not None,
        "classes_loaded": len(class_labels) > 0
    }

@app.get("/classes")
async def get_classes():
    """Get all available disease classes"""
    if not class_labels:
        raise HTTPException(status_code=503, detail="Class labels not loaded")
    
    return {
        "classes": class_labels,
        "total_classes": len(class_labels)
    }

@app.post("/predict")
async def predict_disease(file: UploadFile = File(...)):
    """Predict plant disease from uploaded image"""
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please train the model first.")
    
    if not class_labels:
        raise HTTPException(status_code=503, detail="Class labels not loaded.")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and open image
        image_data = await file.read()
        image = Image.open(BytesIO(image_data))
        
        # Preprocess image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image)
        
        # Check if it's likely a plant image
        if not is_plant_image(predictions[0]):
            return {
                "is_plant": False,
                "message": "This doesn't appear to be a crop or plant image. Please upload an image of crop leaves or plants.",
                "confidence": 0.0,
                "disease": "Not a plant image"
            }
        
        # Get prediction results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = class_labels[predicted_class_idx]
        
        # Format disease name
        formatted_disease = format_disease_name(predicted_class)
        
        # Determine if it's healthy or diseased
        is_healthy = "healthy" in predicted_class.lower()
        
        # Get top 3 predictions for additional info
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = []
        for idx in top_3_indices:
            top_3_predictions.append({
                "disease": format_disease_name(class_labels[idx]),
                "confidence": float(predictions[0][idx])
            })
        
        return {
            "is_plant": True,
            "disease": formatted_disease,
            "confidence": confidence,
            "is_healthy": is_healthy,
            "raw_class": predicted_class,
            "top_predictions": top_3_predictions,
            "total_classes": len(class_labels)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """Predict diseases for multiple images"""
    
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        try:
            # Use the single prediction endpoint logic
            prediction = await predict_disease(file)
            results.append({
                "filename": file.filename,
                "prediction": prediction
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)