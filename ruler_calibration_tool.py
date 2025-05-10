import io
import os 
from google.cloud import vision
import math


try:
    client = vision.ImageAnnotatorClient()
except Exception as e:
    print(f"Error initializing Vision API client: {e}")
    print("Ensure GOOGLE_APPLICATION_CREDENTIALS environment variable is set correctly.")
    client = None # Set to None so functions can check

def get_text_coordinates_from_bytes(image_content_bytes, target_texts):
    """
    Detects text in an image (provided as bytes) and returns the coordinates of target texts.
    Returns a dictionary like: {'text': (center_x, center_y), ...} or None on error.
    """
    if not client:
        print("Vision API client not initialized.")
        return None, "Vision API client not initialized."

    if not image_content_bytes:
        return None, "No image content provided."

    found_coordinates = {}
    detected_texts_info = [] # To store info about all detected texts for debugging

    try:
        image = vision.Image(content=image_content_bytes)
        response = client.text_detection(image=image) # Use the global client

        if response.error.message:
            error_message = (
                f'Vision API Error: {response.error.message}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'
            )
            print(error_message)
            return None, error_message

        texts = response.text_annotations

        if texts:
            # print(f"Found {len(texts)} text elements (raw):") # Debug
            for text_annotation in texts[1:]:  # Skip the first one (full text block)
                desc = text_annotation.description.strip().lower()
                vertices_repr = [(v.x, v.y) for v in text_annotation.bounding_poly.vertices]
                detected_texts_info.append({"description": desc, "vertices": vertices_repr})
                # print(f"- Description: '{desc}', Vertices: {vertices_repr}") # Debug

                for target in target_texts:
                    target_lower = target.lower()
                    # Try exact match or if target is part of the detected text (e.g., "0cm" contains "0")
                    if target_lower == desc or target_lower in desc:
                        x_coords = [vertex.x for vertex in text_annotation.bounding_poly.vertices]
                        y_coords = [vertex.y for vertex in text_annotation.bounding_poly.vertices]
                        center_x = sum(x_coords) / len(x_coords)
                        center_y = sum(y_coords) / len(y_coords)

                        if target not in found_coordinates: # Store the first match found
                            found_coordinates[target] = (center_x, center_y)
                            # print(f"  -> Matched '{target}' at center: ({center_x:.2f}, {center_y:.2f})") # Debug
                        if len(found_coordinates) == len(target_texts): # Optimization
                            break
                if len(found_coordinates) == len(target_texts): # Optimization
                    break
        else:
            print("No text detected in the image by Vision API.")
            return {}, "No text detected in the image." # Return empty dict for coords, specific message

        return found_coordinates, "Text detection complete."

    except Exception as e:
        error_msg = f"Error during text detection: {e}"
        print(error_msg)
        return None, error_msg

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points (x1, y1) and (x2, y2)."""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def calibrate_ruler_from_image_data(image_content_bytes, ruler_actual_length_cm, target_texts_on_ruler):
    """
    Main function to perform ruler calibration from image byte data.

    Args:
        image_content_bytes (bytes): The byte content of the image.
        ruler_actual_length_cm (float): The actual length of the ruler being measured (e.g., 15.0 for a 15cm ruler).
        target_texts_on_ruler (list): A list of two strings representing the start and end markings
                                      on the ruler to find (e.g., ["0", "15"] or ["0cm", "15cm"]).

    Returns:
        dict: A dictionary containing calibration results:
              {
                  "status": "success" or "failure",
                  "message": "Descriptive message",
                  "detected_texts_info": [{"description": "text", "vertices": [(x,y), ...]}, ...], (Optional, for debugging)
                  "found_target_coordinates": {"text_marker": (x, y), ...},
                  "pixel_distance": float or None,
                  "pixels_per_cm": float or None,
                  "pixels_per_inch": float or None
              }
    """
    if not client:
        return {
            "status": "failure",
            "message": "Vision API client not initialized. Cannot proceed.",
            "detected_texts_info": [],
            "found_target_coordinates": {},
            "pixel_distance": None,
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

    if not image_content_bytes:
        return {
            "status": "failure",
            "message": "No image data provided for calibration.",
            "detected_texts_info": [],
            "found_target_coordinates": {},
            "pixel_distance": None,
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

    if not (isinstance(target_texts_on_ruler, list) and len(target_texts_on_ruler) == 2):
        return {
            "status": "failure",
            "message": "target_texts_on_ruler must be a list of two strings (start and end markers).",
            "detected_texts_info": [],
            "found_target_coordinates": {},
            "pixel_distance": None,
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

    print(f"Attempting to find coordinates for: {target_texts_on_ruler}")
    coordinates, text_detection_message = get_text_coordinates_from_bytes(image_content_bytes, target_texts_on_ruler)

    if coordinates is None: # Indicates a significant error in get_text_coordinates_from_bytes
        return {
            "status": "failure",
            "message": f"Could not get text coordinates. Detail: {text_detection_message}",
            "detected_texts_info": [], # Could populate with partial info if available
            "found_target_coordinates": {},
            "pixel_distance": None,
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

    # Check if both target texts were found
    start_marker, end_marker = target_texts_on_ruler[0], target_texts_on_ruler[1]
    if start_marker not in coordinates or end_marker not in coordinates:
        message = (f"Could not find one or both target texts ('{start_marker}', '{end_marker}') with sufficient clarity. "
                   f"Found: {list(coordinates.keys())}. Text detection status: {text_detection_message}")
        # print(message) # Also print to console for immediate feedback during dev
        return {
            "status": "failure",
            "message": message,
            "detected_texts_info": [], # You might want to return detected_texts_info from get_text_coordinates_from_bytes here
            "found_target_coordinates": coordinates,
            "pixel_distance": None,
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

    point_start = coordinates[start_marker]
    point_end = coordinates[end_marker]

    pixel_dist = calculate_distance(point_start, point_end)

    if pixel_dist > 0:
        px_per_cm = pixel_dist / ruler_actual_length_cm
        px_per_inch = px_per_cm * 2.54  # 1 inch = 2.54 cm
        return {
            "status": "success",
            "message": "Calibration successful.",
            # "detected_texts_info": detected_texts_info, # Optionally include all detected texts
            "found_target_coordinates": coordinates,
            "pixel_distance": round(pixel_dist, 2),
            "pixels_per_cm": round(px_per_cm, 2),
            "pixels_per_inch": round(px_per_inch, 2)
        }
    else:
        return {
            "status": "failure",
            "message": "Pixel distance between markers is zero. Cannot calculate scale.",
            # "detected_texts_info": detected_texts_info,
            "found_target_coordinates": coordinates,
            "pixel_distance": round(pixel_dist, 2),
            "pixels_per_cm": None,
            "pixels_per_inch": None
        }

# --- Example of how to use this module (for testing) ---
if __name__ == "__main__":
    print("Testing ruler_calibration_tool.py...")

    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set in your environment
    # For this test, you need an image file named 'ruler_image.jpg' in the same directory,
    # or provide a different path.
    test_image_path = 'ruler_image.jpg' # The same image your original script used
    test_ruler_length = 15.0  # cm
    test_targets = ["0", "15"] # Modify based on your ruler's markings

    if not os.path.exists(test_image_path):
        print(f"Test image '{test_image_path}' not found. Skipping functional test.")
        # You could create dummy bytes for a more basic test:
        # dummy_image_bytes = b"fake image data"
        # results = calibrate_ruler_from_image_data(dummy_image_bytes, test_ruler_length, test_targets)
        # print("\nResults (with dummy data):")
        # import json
        # print(json.dumps(results, indent=2))
    else:
        try:
            with io.open(test_image_path, 'rb') as image_file:
                test_image_content_bytes = image_file.read()
            print(f"Loaded test image '{test_image_path}' ({len(test_image_content_bytes)} bytes)")

            results = calibrate_ruler_from_image_data(test_image_content_bytes, test_ruler_length, test_targets)

            print("\n--- Calibration Results ---")
            import json
            print(json.dumps(results, indent=2))

            if results["status"] == "success":
                print(f"\nSuccessfully calibrated: {results['pixels_per_cm']} pixels/cm")
            else:
                print(f"\nCalibration failed: {results['message']}")

        except Exception as e:
            print(f"Error during test execution: {e}")

    # Test case: No image data
    print("\n--- Testing with no image data ---")
    no_image_results = calibrate_ruler_from_image_data(None, 15.0, ["0", "15"])
    print(json.dumps(no_image_results, indent=2))

    # Test case: Invalid targets
    print("\n--- Testing with invalid target texts ---")
    invalid_targets_results = calibrate_ruler_from_image_data(b"dummy", 15.0, ["0"]) # Not a list of two
    print(json.dumps(invalid_targets_results, indent=2))