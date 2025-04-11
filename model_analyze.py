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
    """Improved skin tone detection that accounts for lighting"""
    try:
        print("  - Using improved skin tone detection...")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Convert to different color spaces
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Create a mask for skin detection
        # This is a more robust method than using fixed thresholds
        lower_hsv = np.array([0, 15, 30], dtype=np.uint8)
        upper_hsv = np.array([50, 170, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        
        # Second range for HSV (for darker skin tones)
        lower_hsv2 = np.array([170, 10, 30], dtype=np.uint8)
        upper_hsv2 = np.array([180, 170, 255], dtype=np.uint8)
        mask_hsv2 = cv2.inRange(hsv_image, lower_hsv2, upper_hsv2)
        
        # YCrCb range for skin detection
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb_image, lower_ycrcb, upper_ycrcb)
        
        # Combine masks
        mask = cv2.bitwise_or(mask_hsv, mask_hsv2)
        mask = cv2.bitwise_and(mask, mask_ycrcb)
        
        # Apply morphological operations to clean the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Extract skin pixels
        skin = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)
        
        # Get mean color of skin pixels
        r, g, b = cv2.mean(skin)[:3]
        h, s, v = cv2.mean(cv2.cvtColor(skin, cv2.COLOR_RGB2HSV))[:3]
        
        # Determine skin tone (1 to 6, where 1 is lightest and 6 is darkest)
        # Using both RGB and HSV values for better accuracy
        # Brightness is a key factor
        brightness = (r + g + b) / 3
        
        # Adjust for lighting conditions based on the overall image brightness
        overall_brightness = np.mean(rgb_image)
        brightness_ratio = brightness / overall_brightness if overall_brightness > 0 else 1
        
        # Adjust brightness based on saturation and value
        adjusted_brightness = brightness * (0.5 + (s / 255) * 0.5) * (0.7 + (v / 255) * 0.3)
        
        # Map to skin tone categories
        if adjusted_brightness > 200:
            skin_tone = 1  # Very light
        elif adjusted_brightness > 170:
            skin_tone = 2  # Light
        elif adjusted_brightness > 140:
            skin_tone = 3  # Medium light
        elif adjusted_brightness > 110:
            skin_tone = 4  # Medium
        elif adjusted_brightness > 80:
            skin_tone = 5  # Medium dark
        else:
            skin_tone = 6  # Dark
        
        print(f"  - Extracted skin color RGB: ({r:.1f}, {g:.1f}, {b:.1f})")
        print(f"  - Adjusted brightness: {adjusted_brightness:.1f}")
        return skin_tone
        
    except Exception as e:
        print(f"  - ERROR in skin tone detection: {str(e)}")
        print(f"  - Traceback: {traceback.format_exc()}")
        # Return a safe middle value instead of exiting
        print("  - Using default skin tone value of 3 (medium)")
        return 3

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for the models"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read image at {image_path}")
            sys.exit(1)
        
        # Resize and normalize
        img = cv2.resize(img, target_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        print(f"  - Traceback: {traceback.format_exc()}")
        sys.exit(1)

def load_model(model_path):
    """Load a saved model with proper error handling"""
    try:
        # Check if the path exists
        if not os.path.exists(model_path):
            print(f"  - Model not found at: {model_path}")
            sys.exit(1)
        
        # Check for saved_model.pb
        saved_model_path = os.path.join(model_path, "saved_model.pb")
        if not os.path.exists(saved_model_path):
            print(f"  - No saved_model.pb found in {model_path}")
            sys.exit(1)
        
        # Load model with TensorFlow's saved_model loader
        print(f"  - Loading model from {model_path}...")
        tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings
        
        try:
            model = tf.saved_model.load(model_path)
            print(f"  - Successfully loaded model")
            return model
        except Exception as e:
            print(f"  - Failed to load model: {str(e)}")
            print(f"  - Traceback: {traceback.format_exc()}")
            sys.exit(1)
    except Exception as e:
        print(f"  - Error loading model: {str(e)}")
        print(f"  - Traceback: {traceback.format_exc()}")
        sys.exit(1)

def run_inference(model, image_tensor, model_name="unknown"):
    """Run inference on a model"""
    try:
        # Get the signature
        infer = model.signatures["serving_default"]
        
        # Run inference
        output = infer(tf.constant(image_tensor, dtype=tf.float32))
        
        # Get output tensor
        output_key = list(output.keys())[0]
        output_value = output[output_key].numpy()
        
        return output_value
        
    except Exception as e:
        print(f"  - ERROR running inference on {model_name} model: {str(e)}")
        print(f"  - Traceback: {traceback.format_exc()}")
        sys.exit(1)

def predict_skin_type(model, image_path):
    """Predict skin type using the model"""
    try:
        # Preprocess image
        img = preprocess_image(image_path)
        
        # Run inference
        output = run_inference(model, img, "skin type")
        
        # Define skin types
        skin_types = ['normal', 'oily', 'dry', 'combination', 'sensitive']
        
        # Get prediction
        if len(output.shape) == 2:
            prediction = np.argmax(output[0])
        else:
            prediction = np.argmax(output)
        
        # Validate prediction
        if prediction >= len(skin_types):
            print(f"  - WARNING: Invalid prediction index {prediction}")
            prediction = 0  # Default to normal
        
        skin_type = skin_types[prediction]
        print(f"  - Predicted skin type: {skin_type}")
        return skin_type
        
    except Exception as e:
        print(f"  - ERROR in skin type prediction: {str(e)}")
        print(f"  - Traceback: {traceback.format_exc()}")
        return "normal"  # Safe default

def detect_acne(model, image_path):
    """Detect acne using the model"""
    try:
        # Preprocess image
        img = preprocess_image(image_path)
        
        # Run inference
        output = run_inference(model, img, "acne")
        
        # Process output
        if len(output.shape) == 2:
            if output.shape[1] == 1:  # Binary classification
                has_acne = output[0][0] > 0.5
                severity = float(output[0][0])
            else:  # Multi-class
                prediction = np.argmax(output[0])
                has_acne = prediction > 0
                severity = float(prediction) / max(1, output.shape[1]-1)
        else:
            if len(output) == 1:  # Binary
                has_acne = output[0] > 0.5
                severity = float(output[0])
            else:  # Multi-class
                prediction = np.argmax(output)
                has_acne = prediction > 0
                severity = float(prediction) / max(1, len(output)-1)
        
        # Map severity to 0-5 scale
        acne_level = int(severity * 5) if has_acne else 0
        
        print(f"  - Acne detected: {'Yes' if has_acne else 'No'} (severity level: {acne_level}/5)")
        return acne_level
        
    except Exception as e:
        print(f"  - ERROR in acne detection: {str(e)}")
        print(f"  - Traceback: {traceback.format_exc()}")
        return 0  # Safe default

def create_feature_vector(skin_type, acne_level, concerns=None):
    """Create feature vector for recommendation"""
    features = [
        'normal', 'dry', 'oily', 'combination', 'sensitive',  # Skin types
        'acne', 'fine lines', 'wrinkles', 'redness',          # Skin concerns
        'dull', 'pore', 'pigmentation', 'blackheads',         # More concerns
        'whiteheads', 'blemishes', 'dark circles', 'eye bags', 'dark spots'
    ]
    
    # Initialize vector
    vector = [0] * len(features)
    
    # Set skin type
    if skin_type in features:
        vector[features.index(skin_type)] = 1
    
    # Set acne level (normalized to 0-1)
    if acne_level > 0:
        acne_idx = features.index('acne')
        vector[acne_idx] = min(1.0, acne_level / 5.0)
    
    # Add additional concerns
    if concerns:
        for concern in concerns:
            if concern in features:
                vector[features.index(concern)] = 1
    
    return vector

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Skin Analysis Using Models')
    parser.add_argument('--image', type=str, help='Path to an existing image file')
    parser.add_argument('--concerns', nargs='+', help='Additional skin concerns (e.g., redness pigmentation)', default=[])
    args = parser.parse_args()
    
    print("Starting skin analysis...")
    
    # Set image path
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
    
    # Load image
    image = load_image(image_path)
    if image is None:
        return
    
    # Display image
    try:
        cv2.imshow("Analysis Image", image)
        cv2.waitKey(2000)
    except Exception as e:
        print(f"Warning: Could not display image: {e}")
    
    # 1. Skin tone detection
    print("\nDetecting skin tone...")
    skin_tone = detect_skin_tone(image_path)
    print(f"Detected skin tone: {skin_tone}/6 (where 1 is lightest, 6 is darkest)")
    
    # 2. Skin type detection
    print("\nAnalyzing skin type...")
    skin_model = load_model(SKIN_MODEL_PATH)
    skin_type = predict_skin_type(skin_model, image_path)
    print(f"Detected skin type: {skin_type}")
    
    # 3. Acne detection
    print("\nDetecting acne...")
    acne_model = load_model(ACNE_MODEL_PATH)
    acne_level = detect_acne(acne_model, image_path)
    print(f"Acne level: {acne_level}/5")
    
    # 4. Generate recommendations
    print("\nGenerating recommendations...")
    
    # Create feature vector
    user_vector = create_feature_vector(
        skin_type=skin_type,
        acne_level=acne_level,
        concerns=args.concerns
    )
    
    print(f"User profile: Skin tone {skin_tone}/6, {skin_type} skin, acne level {acne_level}/5")
    if args.concerns:
        print(f"Additional concerns: {', '.join(args.concerns)}")
    
    # Get recommendations
    try:
        # Convert skin tone to integer (ensure it's an integer)
        skin_tone_int = int(skin_tone)
        
        # Get skincare recommendations
        print("\nGetting skincare recommendations...")
        skincare_recommendations = rec.recs_essentials(vector=user_vector)
        
        # Get makeup recommendations
        print("Getting makeup recommendations...")
        makeup_products = rec.makeup_recommendation(skin_tone_int, skin_type)
        
        # Display recommendations
        print("\nRECOMMENDED SKINCARE PRODUCTS:")
        for category, products in skincare_recommendations.items():
            if products:
                print(f"\n{category.upper()}:")
                for idx, product in enumerate(products[:3]):
                    print(f"- {product['brand']} {product['name']} (${product['price']})")
        
        print("\nRECOMMENDED MAKEUP PRODUCTS FOR SKIN TONE {skin_tone_int}:")
        for product in makeup_products[:5]:
            print(f"- {product['brand']} {product['name']} (${product['price']})")
    
    except Exception as e:
        print(f"ERROR getting recommendations: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 