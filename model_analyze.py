import cv2
import numpy as np
import os
import argparse
import tensorflow as tf
import traceback
import sys
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Fix path issues - add the root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Absolute paths for models and datasets
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
SKIN_TONE_DATASET = os.path.join(ROOT_DIR, "models/skin_tone/skin_tone_dataset.csv")
SKIN_MODEL_PATH = os.path.join(ROOT_DIR, "models/skin_model")
ACNE_MODEL_PATH = os.path.join(ROOT_DIR, "models/acne_model")

# Add the models directory to sys.path
models_dir = os.path.join(ROOT_DIR, "models")
if models_dir not in sys.path:
    sys.path.append(models_dir)

# Add the recommender directory to sys.path
recommender_dir = os.path.join(ROOT_DIR, "models/recommender")
if recommender_dir not in sys.path:
    sys.path.append(recommender_dir)

# Import directly from the modules
print("\nImporting modules...")
try:
    from models.skin_tone.skin_detection import skin_detection
    from models.skin_tone.skin_tone_knn import identify_skin_tone
    print("Successfully imported skin tone modules")
except Exception as e:
    print(f"ERROR importing skin tone modules: {str(e)}")
    sys.exit(1)  # Exit if essential modules can't be imported

try:
    # Directly import from the recommender module
    sys.path.append(os.path.join(ROOT_DIR, "models/recommender"))
    # Import the module directly
    import rec
    print("Successfully imported recommendation modules")
except Exception as e:
    print(f"ERROR importing recommendation modules: {str(e)}")
    print(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)  # Exit if essential modules can't be imported

def load_image(image_path):
    """Load an image and return it, or None if it can't be read"""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    return img

def detect_skin_tone(image_path):
    """Use the histogram-based thresholding and k-means clustering method to detect skin tone"""
    try:
        print("  - Using skin tone KNN model...")
        
        # Use the hardcoded absolute path to the dataset
        if os.path.exists(SKIN_TONE_DATASET):
            dataset_path = SKIN_TONE_DATASET
            print(f"  - Using dataset at: {dataset_path}")
        else:
            print(f"  - Dataset not found at {SKIN_TONE_DATASET}, trying relative path...")
            relative_path = "models/skin_tone/skin_tone_dataset.csv"
            if os.path.exists(relative_path):
                dataset_path = relative_path
                print(f"  - Using relative path: {dataset_path}")
            else:
                raise FileNotFoundError(f"Cannot find dataset file")
                
        # Call the skin detection function directly
        print("  - Detecting skin...")
        mean_color = skin_detection(image_path)
        print(f"  - Extracted skin color: {mean_color}")
        
        # Get skin tone
        skin_tone = identify_skin_tone(image_path, dataset_path)
        print(f"  - Identified skin tone: {skin_tone}")
        return skin_tone
        
    except Exception as e:
        print(f"  - ERROR in skin tone detection: {str(e)}")
        print(f"  - Traceback: {traceback.format_exc()}")
        print("Exiting due to error in skin tone detection")
        sys.exit(1)

def preprocess_image_for_efficientnet(image_path, target_size=(224, 224)):
    """Preprocess image specifically for EfficientNet models"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            sys.exit(1)
        
        # Resize and convert from BGR to RGB (EfficientNet expects RGB)
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        print(f"  - Traceback: {traceback.format_exc()}")
        sys.exit(1)

def load_efficientnet_model(model_path):
    """Load a saved EfficientNet model with proper error handling"""
    try:
        # Check if the path exists
        if not os.path.exists(model_path):
            print(f"  - Model not found at: {model_path}")
            sys.exit(1)
        
        # Check if saved_model.pb exists
        saved_model_path = os.path.join(model_path, "saved_model.pb")
        if not os.path.exists(saved_model_path):
            print(f"  - No saved_model.pb found in {model_path}")
            sys.exit(1)
        
        # Load model with TensorFlow's saved_model loader
        print(f"  - Loading model from {model_path}...")
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
        
        try:
            # Load model and capture output
            model = tf.saved_model.load(model_path)
            print(f"  - Successfully loaded model")
            
            # Print model info for debugging
            print(f"  - Model signatures: {list(model.signatures.keys())}")
            return model
        except Exception as e:
            print(f"  - Failed to load model: {str(e)}")
            print(f"  - Traceback: {traceback.format_exc()}")
            sys.exit(1)
    except Exception as e:
        print(f"  - Error in load_efficientnet_model: {str(e)}")
        print(f"  - Traceback: {traceback.format_exc()}")
        sys.exit(1)

def run_model_inference(model, image_tensor, model_name="unknown"):
    """Generic function to run inference on any TensorFlow model with error handling"""
    try:
        # Get available signatures
        signatures = list(model.signatures.keys())
        if not signatures:
            print(f"  - ERROR: No signatures found in {model_name} model")
            sys.exit(1)
        
        # Default to serving_default or use the first available signature
        signature = "serving_default" if "serving_default" in signatures else signatures[0]
        print(f"  - Using signature: {signature}")
        
        # Run inference
        result = model.signatures[signature](tf.constant(image_tensor, dtype=tf.float32))
        print(f"  - Inference result keys: {list(result.keys())}")
        
        # Find the output tensor (look for common output names)
        output_key = None
        for key in result.keys():
            if any(term in key.lower() for term in ['dense', 'logits', 'output', 'prediction', 'softmax', 'prob']):
                output_key = key
                break
        
        # If no known output key found, use the first one
        if output_key is None and result:
            output_key = list(result.keys())[0]
            
        if output_key:
            print(f"  - Using output key: {output_key}")
            output = result[output_key].numpy()
            print(f"  - Output shape: {output.shape}, values: {output}")
            return output
        else:
            print(f"  - ERROR: Could not find output tensor in {model_name} model")
            sys.exit(1)
    except Exception as e:
        print(f"  - ERROR running inference on {model_name} model: {str(e)}")
        print(f"  - Traceback: {traceback.format_exc()}")
        sys.exit(1)

def predict_skin_type(model, image_path):
    """Use the EfficientNet model to predict skin type"""
    try:
        # Preprocess the image
        print("  - Preprocessing image for skin type model...")
        img = preprocess_image_for_efficientnet(image_path)
        
        # Run inference
        print("  - Running skin type model inference...")
        output = run_model_inference(model, img, "skin type")
        
        if output is None:
            print("  - Failed to get output from skin type model")
            sys.exit(1)
        
        # Define the skin types
        skin_types = ['normal', 'dry', 'oily', 'combination', 'sensitive']
        
        # Handle different output shapes
        if len(output.shape) == 1:
            # Single dimension output
            skin_type_idx = np.argmax(output)
        elif len(output.shape) == 2:
            # Batched output
            skin_type_idx = np.argmax(output[0])
        else:
            print(f"  - Unexpected output shape: {output.shape}")
            skin_type_idx = 0  # Default to 'normal'
        
        # Check if the index is within bounds
        if skin_type_idx >= len(skin_types):
            print(f"  - WARNING: Index {skin_type_idx} out of bounds for skin types")
            skin_type_idx = 0  # Default to 'normal'
        
        skin_type = skin_types[skin_type_idx]
        print(f"  - Predicted skin type: {skin_type} (index {skin_type_idx})")
        return skin_type
    except Exception as e:
        print(f"  - ERROR in skin type prediction: {str(e)}")
        print(f"  - Traceback: {traceback.format_exc()}")
        sys.exit(1)

def detect_acne(model, image_path):
    """Use the EfficientNet model to detect acne"""
    try:
        # Preprocess the image
        print("  - Preprocessing image for acne model...")
        img = preprocess_image_for_efficientnet(image_path)
        
        # Run inference
        print("  - Running acne model inference...")
        output = run_model_inference(model, img, "acne")
        
        if output is None:
            print("  - Failed to get output from acne model")
            sys.exit(1)
        
        # Handle different output shapes
        if len(output.shape) == 0:
            # Single scalar output (binary)
            acne_value = 1 if output > 0.5 else 0
        elif len(output.shape) == 1:
            # 1D output (might be binary or multiclass)
            if output.shape[0] == 1:
                # Single value binary classification
                acne_value = 1 if output[0] > 0.5 else 0
            else:
                # Multiclass classification
                acne_value = np.argmax(output)
        elif len(output.shape) == 2:
            # Batched output
            if output.shape[1] == 1:
                # Binary classification
                acne_value = 1 if output[0][0] > 0.5 else 0
            else:
                # Multiclass
                acne_value = np.argmax(output[0])
        else:
            print(f"  - Unexpected output shape: {output.shape}")
            acne_value = 0  # Default to no acne
        
        has_acne = acne_value > 0
        print(f"  - Model detected acne: {'Yes' if has_acne else 'No'} (value: {acne_value})")
        return acne_value, 0.0
    except Exception as e:
        print(f"  - ERROR in acne detection: {str(e)}")
        print(f"  - Traceback: {traceback.format_exc()}")
        sys.exit(1)

def create_vector_for_recommender(skin_type, has_acne):
    """Create a feature vector for the recommender system in the correct format"""
    # Create a feature vector that matches the recommender's expected format
    features = ['normal', 'dry', 'oily', 'combination', 'acne', 'sensitive', 'fine lines', 'wrinkles', 'redness',
            'dull', 'pore', 'pigmentation', 'blackheads', 'whiteheads', 'blemishes', 'dark circles', 'eye bags', 'dark spots']
    
    # Initialize vector
    vector = [0] * len(features)
    
    # Set skin type
    if skin_type in features:
        vector[features.index(skin_type)] = 1
    
    # Set acne if detected
    if has_acne:
        vector[features.index('acne')] = 1
    
    print(f"Created feature vector: {vector}")
    return vector

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Skin Analysis Using Models')
    parser.add_argument('--image', type=str, help='Path to an existing image file')
    args = parser.parse_args()
    
    print("Starting skin analysis with models...")
    
    # Verify that recommendation system can be accessed
    print("\nVerifying recommender system...")
    try:
        # Check if the required CSV files exist
        recommender_dir = os.path.join(ROOT_DIR, 'models/recommender')
        for csv_file in ['final.csv', 'makeup_final.csv']:
            csv_path = os.path.join(recommender_dir, csv_file)
            if not os.path.exists(csv_path):
                print(f"ERROR: Required recommender file not found: {csv_path}")
                sys.exit(1)
        print("Recommender system files verified.")
    except Exception as e:
        print(f"ERROR verifying recommender: {e}")
        sys.exit(1)
    
    # If no image is provided, look for sample_face.png
    if not args.image:
        if os.path.exists("sample_face.png"):
            image_path = "sample_face.png"
            print(f"No image provided, using sample image: {image_path}")
        else:
            print("ERROR: No image provided and sample_face.png not found.")
            print("Please provide an image with: python model_analyze.py --image path/to/image.png")
            return
    else:
        image_path = args.image
    
    # Load the image
    image = load_image(image_path)
    if image is None:
        return
    
    # Display the image
    try:
        cv2.imshow("Analysis Image", image)
        cv2.waitKey(2000)  # Show for 2 seconds
    except Exception as e:
        print(f"Warning: Could not display image: {e}")
    
    # Analyze skin
    print("\nAnalyzing skin using models...\n")
    
    # 1. Get skin tone using the histogram-based approach
    print("Detecting skin tone...")
    skin_tone = detect_skin_tone(image_path)
    print(f"Detected skin tone: {skin_tone}/6 (where 1 is lightest, 6 is darkest)")
    
    # 2. Load and use the skin type model (EfficientNet)
    print("\nAnalyzing skin type...")
    skin_model = load_efficientnet_model(SKIN_MODEL_PATH)
    skin_type = predict_skin_type(skin_model, image_path)
    print(f"Detected skin type: {skin_type}")
    
    # 3. Load and use the acne detection model (EfficientNet)
    print("\nDetecting acne...")
    acne_model = load_efficientnet_model(ACNE_MODEL_PATH)
    acne_value, acne_percent = detect_acne(acne_model, image_path)
    has_acne = bool(acne_value)
    print(f"Acne detected: {'Yes' if has_acne else 'No'}")
    
    # 4. Get recommendations from the recommender system
    print("\nGenerating recommendations from recommender system...")
    
    # Create a feature vector for the recommender
    user_vector = create_vector_for_recommender(skin_type, has_acne)
    
    # Get skincare recommendations
    print("Getting skincare recommendations...")
    try:
        # Convert skin tone to integer if it's a float
        skin_tone_int = int(skin_tone) if isinstance(skin_tone, (int, float)) else skin_tone
        
        # Call recommender functions directly from the module
        skincare_recommendations = rec.recs_essentials(vector=user_vector)
        print("Successfully received skincare recommendations")
        
        # Get makeup recommendations
        print("Getting makeup recommendations...")
        makeup_products = rec.makeup_recommendation(skin_tone_int, skin_type)
        print("Successfully received makeup recommendations")
        
        # Display recommendations
        print("\nRECOMMENDED SKINCARE PRODUCTS:")
        for category, products in skincare_recommendations.items():
            if products:
                print(f"\n{category.upper()}:")
                for product in products[:2]:  # Show just top 2 products per category
                    print(f"- {product['brand']} {product['name']} (${product['price']})")
        
        print("\nRECOMMENDED MAKEUP PRODUCTS:")
        for product in makeup_products[:5]:
            print(f"- {product['brand']} {product['name']} (${product['price']})")
    
    except Exception as e:
        print(f"ERROR getting recommendations: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)  # Exit with error instead of falling back
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 