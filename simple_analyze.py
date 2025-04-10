import cv2
import numpy as np
import os
import argparse
import random
from pathlib import Path

# Skin tone labels
SKIN_TONES = [1, 2, 3, 4, 5, 6]
SKIN_TYPES = ['normal', 'dry', 'oily', 'combination', 'sensitive']

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

def analyze_skin_tone(image):
    """Analyze skin tone using a simplified approach"""
    # In a real app, this would use your skin tone model
    # For this simple version, we'll use color averaging to estimate tone
    
    # Convert to the right color space and get average values
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_img)
    
    # Take the middle region of the image as likely face area
    height, width = image.shape[:2]
    face_region = hsv_img[height//4:3*height//4, width//4:3*width//4]
    
    # Calculate average color values
    avg_h = np.mean(face_region[:,:,0])
    avg_s = np.mean(face_region[:,:,1])
    avg_v = np.mean(face_region[:,:,2])
    
    # Simple mapping of value (brightness) to skin tone
    # Lower values (darker) = higher tone number
    # This is extremely simplified compared to your actual model
    if avg_v > 200:
        skin_tone = 1  # Very light
    elif avg_v > 180:
        skin_tone = 2  # Light
    elif avg_v > 160:
        skin_tone = 3  # Medium light
    elif avg_v > 140:
        skin_tone = 4  # Medium
    elif avg_v > 120:
        skin_tone = 5  # Medium dark
    else:
        skin_tone = 6  # Dark
    
    return skin_tone, (avg_h, avg_s, avg_v)

def analyze_skin_type(image):
    """Determine skin type based on image analysis"""
    # In a real app, this would use your skin type model
    # For this simple version, we use color/texture approximation
    
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Get the middle region of the image (face area)
    height, width = image.shape[:2]
    face_region = image[height//4:3*height//4, width//4:3*width//4]
    face_hsv = hsv_img[height//4:3*height//4, width//4:3*width//4]
    face_ycrcb = ycrcb_img[height//4:3*height//4, width//4:3*width//4]
    
    # Calculate standard deviation of saturation as a measure of skin evenness
    s_std = np.std(face_hsv[:,:,1])
    
    # Calculate average luminance
    avg_y = np.mean(face_ycrcb[:,:,0])
    
    # Simplified logic for skin type determination
    if s_std > 35:
        skin_type = 'combination'  # High variance in color
    elif avg_y < 130:
        skin_type = 'dry'  # Darker/duller complexion
    elif avg_y > 180:
        skin_type = 'oily'  # Brighter/shinier complexion
    else:
        skin_type = 'normal'  # Middle values
    
    return skin_type

def detect_acne(image):
    """Detect presence of acne in the image"""
    # In a real app, this would use your acne model
    # For this simple version, we'll use color analysis to estimate
    
    # Convert to the right color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Focus on the middle part of the image (face area)
    height, width = image.shape[:2]
    face_region = hsv_img[height//4:3*height//4, width//4:3*width//4]
    
    # Look for reddish hues that might indicate acne
    # Simplified thresholds for red/pink tones
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red regions
    mask1 = cv2.inRange(face_region, lower_red, upper_red)
    mask2 = cv2.inRange(face_region, lower_red2, upper_red2)
    
    # Combine masks
    red_mask = mask1 + mask2
    
    # Calculate percentage of pixels that are in the red range
    red_pixel_percent = np.sum(red_mask > 0) / (face_region.shape[0] * face_region.shape[1])
    
    # If more than 5% of pixels are reddish, consider it as having acne
    has_acne = red_pixel_percent > 0.05
    
    return has_acne, red_pixel_percent

def get_skincare_recommendations(skin_type, has_acne):
    """Return skincare product recommendations based on analysis"""
    print("\nRECOMMENDED SKINCARE PRODUCTS:")
    
    # Cleansers
    print("\nCLEANSERS:")
    if skin_type == 'oily' or has_acne:
        print("- CeraVe Foaming Facial Cleanser ($15)")
        print("- La Roche-Posay Effaclar Purifying Foaming Gel ($23)")
    elif skin_type == 'dry':
        print("- CeraVe Hydrating Facial Cleanser ($15)")
        print("- Neutrogena Hydro Boost Hydrating Cleansing Gel ($12)")
    else:
        print("- Cetaphil Gentle Skin Cleanser ($14)")
        print("- Kiehl's Ultra Facial Cleanser ($22)")
    
    # Moisturizers
    print("\nMOISTURIZERS:")
    if skin_type == 'oily':
        print("- Neutrogena Hydro Boost Water Gel ($24)")
        print("- La Roche-Posay Effaclar Mat ($32)")
    elif skin_type == 'dry':
        print("- CeraVe Moisturizing Cream ($19)")
        print("- First Aid Beauty Ultra Repair Cream ($34)")
    elif skin_type == 'combination':
        print("- Clinique Dramatically Different Moisturizing Gel ($30)")
        print("- Belif The True Cream Aqua Bomb ($38)")
    else:
        print("- Neutrogena Oil-Free Moisture ($12)")
        print("- Kiehl's Ultra Facial Cream ($32)")
    
    # Treatments
    print("\nTREATMENTS:")
    if has_acne:
        print("- Paula's Choice 2% BHA Liquid Exfoliant ($30)")
        print("- The Ordinary Niacinamide 10% + Zinc 1% ($6)")
    elif skin_type == 'dry':
        print("- The Ordinary Hyaluronic Acid 2% + B5 ($7)")
        print("- Fresh Rose Deep Hydration Face Cream ($42)")
    elif skin_type == 'combination':
        print("- The Ordinary Azelaic Acid Suspension 10% ($8)")
        print("- Sunday Riley Good Genes All-In-One Lactic Acid Treatment ($85)")
    else:
        print("- Drunk Elephant C-Firma Vitamin C Day Serum ($80)")
        print("- The Ordinary Buffet ($15)")

def get_makeup_recommendations(skin_tone, skin_type):
    """Return makeup product recommendations based on analysis"""
    print("\nRECOMMENDED MAKEUP PRODUCTS:")
    
    # Foundation
    print("\nFOUNDATION:")
    if skin_type == 'oily':
        print(f"- Fenty Beauty Pro Filt'r Soft Matte Foundation (Shade {100 + skin_tone*50}) ($36)")
        print(f"- Estée Lauder Double Wear Stay-in-Place Foundation (Shade {skin_tone}.0) ($43)")
    elif skin_type == 'dry':
        print(f"- NARS Sheer Glow Foundation (Shade {['Mont Blanc', 'Punjab', 'Syracuse', 'Tahoe', 'Macao', 'Trinidad'][skin_tone-1]}) ($47)")
        print(f"- Giorgio Armani Luminous Silk Foundation (Shade {skin_tone}.5) ($64)")
    else:
        print(f"- MAC Studio Fix Fluid SPF 15 (Shade NC{15 + skin_tone*5}) ($33)")
        print(f"- Make Up For Ever HD Skin Foundation (Shade Y{200 + skin_tone*50}) ($43)")
    
    # Concealer
    print("\nCONCEALER:")
    print(f"- NARS Radiant Creamy Concealer (Shade {['Chantilly', 'Vanilla', 'Custard', 'Ginger', 'Caramel', 'Amande'][skin_tone-1]}) ($30)")
    print(f"- Tarte Shape Tape Concealer (Shade {10 + skin_tone*5}N) ($27)")
    
    # Blush
    print("\nBLUSH:")
    if skin_tone <= 2:
        print("- NARS Blush in Orgasm ($30)")
        print("- Benefit Cosmetics Dandelion Blush ($30)")
    elif skin_tone <= 4:
        print("- NARS Blush in Deep Throat ($30)")
        print("- Milani Baked Blush in Luminoso ($9)")
    else:
        print("- NARS Blush in Exhibit A ($30)")
        print("- Fenty Beauty Cheeks Out Freestyle Cream Blush in Daiquiri Dip ($20)")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Simple Skin Analysis and Product Recommendation')
    parser.add_argument('--image', type=str, help='Path to an existing image file')
    args = parser.parse_args()
    
    print("Starting simplified skin analysis system...")
    
    # If no image is provided, look for sample_face.jpg
    if not args.image:
        if os.path.exists("sample_face.png"):
            image_path = "sample_face.png"
            print(f"No image provided, using sample image: {image_path}")
        else:
            print("ERROR: No image provided and sample_face.png not found.")
            print("Please either:")
            print("1. Create a sample image with: python create_test_image.py")
            print("2. Provide an image path with: python simple_analyze.py --image path/to/image.png")
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
    print("\nAnalyzing skin...\n")
    
    # Get skin tone
    skin_tone, hsv_avg = analyze_skin_tone(image)
    print(f"Detected skin tone: {skin_tone}/6 (where 1 is lightest, 6 is darkest)")
    
    # Get skin type
    skin_type = analyze_skin_type(image)
    print(f"Detected skin type: {skin_type}")
    
    # Check for acne
    has_acne, acne_percent = detect_acne(image)
    print(f"Acne detected: {'Yes' if has_acne else 'No'} ({acne_percent*100:.1f}% redness)")
    
    # Get product recommendations
    get_skincare_recommendations(skin_type, has_acne)
    get_makeup_recommendations(skin_tone, skin_type)
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 