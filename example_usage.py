#!/usr/bin/env python3
"""
HaMeR Usage Examples

This script demonstrates different ways to use HaMeR for 3D hand reconstruction:
1. Direct library interface (no server needed)
2. API server interface (requires running server)
"""

import cv2
import numpy as np
from pathlib import Path
import base64
import requests
import json


def example_library_usage():
    """Examples of using HaMeR as a library directly"""
    print("📚 Library Usage Examples")
    print("=" * 50)

    # Import the HaMeR inference module
    from hamer.inference import predict, HaMeRInference

    # Example 1: Process a directory of images
    print("\n1️⃣ Processing a directory of images:")
    try:
        result = predict(
            image_directory="example_data",
            file_extensions=['*.jpg', '*.png', '*.jpeg'],
            rescale_factor=2.0,
            body_detector='vitdet'
        )
        print(f"   ✅ Processed {result['total_images']} images")
        print(f"   ✅ {result['successful_predictions']} successful predictions")

        # Access results
        for img_result in result['results'][:2]:  # Show first 2 results
            print(f"   📄 {img_result['image_name']}: {len(img_result['hands'])} hands detected")

    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Example 2: Process a list of image files
    print("\n2️⃣ Processing a list of image files:")
    try:
        image_files = ["example_data/test1.jpg"]  # Add more files as needed

        if Path(image_files[0]).exists():
            result = predict(
                images=image_files,
                image_names=["my_test_image.jpg"],
                rescale_factor=2.0,
                body_detector='vitdet'
            )
            print(f"   ✅ Processed {result['total_images']} images")
            print(f"   ✅ {result['successful_predictions']} successful predictions")

            # Access hand data
            if result['results'] and result['results'][0]['hands']:
                hand = result['results'][0]['hands'][0]
                hand_type = "Right" if hand['is_right_hand'] else "Left"
                print(f"   🖐️ Found {hand_type} hand with {len(hand['vertices'])} vertices")

    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Example 3: Process numpy arrays directly
    print("\n3️⃣ Processing numpy arrays directly:")
    try:
        test_image_path = "example_data/test1.jpg"
        if Path(test_image_path).exists():
            # Load image as numpy array
            img_array = cv2.imread(test_image_path)

            result = predict(
                images=[img_array],  # Pass numpy array directly
                image_names=["numpy_image.jpg"],
                rescale_factor=2.0
            )
            print(f"   ✅ Processed numpy array: {result['successful_predictions']} successful predictions")

    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Example 4: Using HaMeRInference class directly for more control
    print("\n4️⃣ Using HaMeRInference class for advanced usage:")
    try:
        # Create inference engine with specific device
        engine = HaMeRInference(device='cuda')  # or 'cpu'

        # Change body detector
        engine.set_body_detector('vitdet')

        # Process images
        result = engine.predict(
            image_directory="example_data",
            rescale_factor=1.5,  # Different rescale factor
            body_detector='vitdet'
        )
        print(f"   ✅ Advanced usage: {result['successful_predictions']} successful predictions")

    except Exception as e:
        print(f"   ❌ Failed: {e}")


def example_api_usage():
    """Examples of using HaMeR via API (requires running server)"""
    print("\n🌐 API Usage Examples")
    print("=" * 50)

    base_url = "http://localhost:8000"

    # Check if API server is running
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code != 200:
            print("   ⚠️ API server not running. Start with: python hamer_api.py")
            return
    except requests.RequestException:
        print("   ⚠️ API server not running. Start with: python hamer_api.py")
        return

    print("   ✅ API server is running")

    # Example 1: API directory processing
    print("\n1️⃣ API Directory Processing:")
    try:
        request_data = {
            "image_directory": "example_data",
            "file_extensions": ["*.jpg", "*.png", "*.jpeg"],
            "rescale_factor": 2.0,
            "body_detector": "vitdet"
        }

        response = requests.post(f"{base_url}/predict_from_directory",
                               json=request_data, timeout=60)

        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ API processed {result['total_images']} images")
            print(f"   ✅ {result['successful_predictions']} successful predictions")
        else:
            print(f"   ❌ API request failed: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Example 2: API base64 image processing
    print("\n2️⃣ API Base64 Image Processing:")
    try:
        test_image_path = "example_data/test1.jpg"
        if Path(test_image_path).exists():
            # Encode image to base64
            with open(test_image_path, "rb") as f:
                image_data = f.read()
                image_b64 = base64.b64encode(image_data).decode("utf-8")

            request_data = {
                "images_base64": [image_b64],
                "image_names": ["api_test_image.jpg"],
                "rescale_factor": 2.0,
                "body_detector": "vitdet"
            }

            response = requests.post(f"{base_url}/predict_from_images",
                                   json=request_data, timeout=60)

            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ API processed {result['total_images']} base64 images")
                print(f"   ✅ {result['successful_predictions']} successful predictions")
            else:
                print(f"   ❌ API request failed: {response.status_code}")

    except Exception as e:
        print(f"   ❌ Failed: {e}")


def example_error_handling():
    """Examples of error handling"""
    print("\n🛡️ Error Handling Examples")
    print("=" * 50)

    from hamer.inference import predict

    # Example 1: Invalid input validation
    print("\n1️⃣ Input validation:")
    try:
        result = predict(
            image_directory="example_data",
            images=["test.jpg"]  # Both provided - should fail
        )
    except ValueError as e:
        print(f"   ✅ Correctly caught error: {e}")

    # Example 2: Missing directory
    print("\n2️⃣ Missing directory:")
    try:
        result = predict(image_directory="nonexistent_directory")
    except ValueError as e:
        print(f"   ✅ Correctly caught error: {e}")

    # Example 3: Invalid image
    print("\n3️⃣ Invalid image handling:")
    try:
        result = predict(
            images=["invalid_base64_string"],
            image_names=["invalid.jpg"]
        )
        # Check if it handled the error gracefully
        if result['failed_predictions'] > 0:
            print(f"   ✅ Gracefully handled invalid image: {result['failed_predictions']} failed")
    except Exception as e:
        print(f"   ⚠️ Exception (expected): {e}")


def main():
    """Run all examples"""
    print("🚀 HaMeR Usage Examples")
    print("🎯 This demonstrates the modular design:")
    print("   • Library interface: Direct Python usage (no server needed)")
    print("   • API interface: HTTP REST API (requires server)")
    print("   • Both use the same underlying inference engine!")

    # Library examples
    example_library_usage()

    # API examples (if server is running)
    example_api_usage()

    # Error handling examples
    example_error_handling()

    print("\n" + "=" * 50)
    print("📋 Summary:")
    print("   ✅ Library interface: import hamer.inference; predict(...)")
    print("   ✅ API interface: POST requests to /predict_from_directory or /predict_from_images")
    print("   ✅ Same core functionality, different interfaces!")
    print("   ✅ Choose library for direct usage, API for web services")


if __name__ == "__main__":
    main()