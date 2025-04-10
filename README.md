# Skin Analysis and Product Recommendation System

This application analyzes your skin using three different models and recommends skincare and makeup products based on the analysis.

## Features

- **Skin Tone Analysis**: Detects your skin tone using computer vision techniques
- **Skin Type Classification**: Identifies your skin type (normal, dry, oily, combination, sensitive)
- **Acne Detection**: Detects the presence of acne
- **Product Recommendations**: Suggests skincare and makeup products based on the analysis

## Requirements

- Python 3.8+ (Python 3.12+ recommended)
- Webcam (optional - only needed for live capture)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Test if all models are available and working:
   ```
   python test_models.py
   ```

## Usage

### Option 1: Using your webcam

Run the main application without any arguments to capture a photo from your webcam:

```
python main.py
```

The application will:
1. Capture a photo using your webcam (after a 3-second countdown)
2. Display the captured photo
3. Analyze your skin tone, skin type, and detect acne
4. Recommend skincare and makeup products based on the analysis
5. Clean up temporary files when done

### Option 2: Using an existing image

If you already have a photo or are having trouble with the webcam, you can use an existing image file:

```
python main.py --image /path/to/your/photo.jpg
```

### Option 3: Using a generated test image

If you don't have a suitable image, you can generate a test image:

```
python create_test_image.py
```

This will create a `sample_face.jpg` file that you can use for testing:

```
python main.py --image sample_face.jpg
```

## Models

The application uses the following models:
- Skin Tone Detection: KNN-based classifier
- Skin Type Classification: Deep learning model
- Acne Detection: Deep learning model

## Recommendations

The system provides recommendations for:
- Various skincare product categories
- Makeup products matching your skin tone and type

## Troubleshooting

If you encounter issues:
1. Run the test script to verify models are working correctly:
   ```
   python test_models.py
   ```
2. If you're having trouble with dependencies:
   - Try using Python 3.12+ 
   - Try installing dependencies one by one
3. If you're having issues with the webcam:
   - Make sure your camera is properly connected
   - Use the --image option with an existing photo
   - Use the generated test image by running `create_test_image.py` 