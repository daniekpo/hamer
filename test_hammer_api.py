#!/usr/bin/env python3
"""
Test script for HaMeR Hand Tracking API

This script tests both endpoints of the API with example data.
"""

import base64
import json
import os
import requests
import time
from pathlib import Path


def test_health_check(base_url: str = "http://localhost:8000"):
    """Test the health check endpoint"""
    print("🏥 Testing health check endpoint...")

    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Health check passed: {result}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_directory_endpoint(base_url: str = "http://localhost:8000",
                          image_dir: str = "./hamer/example_data"):
    """Test the directory prediction endpoint"""
    print("📁 Testing directory endpoint...")

    # Check if directory exists
    if not Path(image_dir).exists():
        print(f"❌ Directory not found: {image_dir}")
        return False

    request_data = {
        "image_directory": image_dir,
        "file_extensions": ["*.jpg", "*.png", "*.jpeg"],
        "rescale_factor": 2.0,
        "body_detector": "vitdet"
    }

    try:
        print(f"   Sending request to process images in: {image_dir}")
        response = requests.post(
            f"{base_url}/predict_from_directory",
            json=request_data,
            timeout=120  # Longer timeout for model processing
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Directory prediction successful!")
            print(f"   Total images: {result['total_images']}")
            print(f"   Successful: {result['successful_predictions']}")
            print(f"   Failed: {result['failed_predictions']}")

                        # Print details for first few results
            for i, img_result in enumerate(result['results'][:3]):
                print(f"   Image {i+1}: {img_result['image_name']}")
                print(f"     Hands detected: {len(img_result['hands'])}")

                for j, hand in enumerate(img_result['hands']):
                    hand_type = "Right" if hand['is_right_hand'] else "Left"
                    print(f"       Hand {j+1}: {hand_type} hand")
                    print(f"         Vertices: {len(hand['vertices'])} points")
                    print(f"         Joints: {len(hand['joints'])} points")

                    # Validate prediction structure
                    is_valid, message = validate_prediction_structure(hand)
                    if is_valid:
                        print(f"         ✅ Structure validation: {message}")
                    else:
                        print(f"         ❌ Structure validation failed: {message}")
                        return False

            return True
        else:
            print(f"❌ Directory prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.RequestException as e:
        print(f"❌ Directory prediction failed: {e}")
        return False


def test_image_list_endpoint(base_url: str = "http://localhost:8000",
                           test_image_path: str = "./hamer/example_data/test1.jpg"):
    """Test the image list prediction endpoint"""
    print("🖼️ Testing image list endpoint...")

    # Check if test image exists
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False

    # Encode image to base64
    try:
        with open(test_image_path, "rb") as f:
            image_data = f.read()
            image_b64 = base64.b64encode(image_data).decode('utf-8')
    except Exception as e:
        print(f"❌ Failed to encode image: {e}")
        return False

    request_data = {
        "images_base64": [image_b64],
        "image_names": [Path(test_image_path).name],
        "rescale_factor": 2.0,
        "body_detector": "vitdet"
    }

    try:
        print(f"   Sending base64 encoded image: {Path(test_image_path).name}")
        response = requests.post(
            f"{base_url}/predict_from_images",
            json=request_data,
            timeout=120  # Longer timeout for model processing
        )

        if response.status_code == 200:
            result = response.json()
            print(f"✅ Image list prediction successful!")
            print(f"   Total images: {result['total_images']}")
            print(f"   Successful: {result['successful_predictions']}")
            print(f"   Failed: {result['failed_predictions']}")

            # Print details for the result
            if result['results']:
                img_result = result['results'][0]
                print(f"   Image: {img_result['image_name']}")
                print(f"     Hands detected: {len(img_result['hands'])}")

                for j, hand in enumerate(img_result['hands']):
                    hand_type = "Right" if hand['is_right_hand'] else "Left"
                    print(f"       Hand {j+1}: {hand_type} hand")
                    print(f"         Vertices: {len(hand['vertices'])} points")
                    print(f"         Joints: {len(hand['joints'])} points")
                    print(f"         Confidence: {hand['confidence_score']}")

                    # Validate prediction structure
                    is_valid, message = validate_prediction_structure(hand)
                    if is_valid:
                        print(f"         ✅ Structure validation: {message}")
                    else:
                        print(f"         ❌ Structure validation failed: {message}")
                        return False

            return True
        else:
            print(f"❌ Image list prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False

    except requests.RequestException as e:
        print(f"❌ Image list prediction failed: {e}")
        return False


def validate_prediction_structure(hand_prediction: dict):
    """Validate the structure of a hand prediction"""
    required_fields = ['vertices', 'joints', 'is_right_hand', 'confidence_score']

    for field in required_fields:
        if field not in hand_prediction:
            return False, f"Missing field: {field}"

    # Check vertices structure (should be 778x3)
    vertices = hand_prediction['vertices']
    if not isinstance(vertices, list) or len(vertices) != 778:
        return False, f"Invalid vertices structure: expected 778 points, got {len(vertices)}"

    if len(vertices) > 0 and len(vertices[0]) != 3:
        return False, f"Invalid vertex dimension: expected 3D points, got {len(vertices[0])}D"

    # Check joints structure (should be 21x3)
    joints = hand_prediction['joints']
    if not isinstance(joints, list) or len(joints) != 21:
        return False, f"Invalid joints structure: expected 21 points, got {len(joints)}"

    if len(joints) > 0 and len(joints[0]) != 3:
        return False, f"Invalid joint dimension: expected 3D points, got {len(joints[0])}D"

    return True, "Valid structure"


def main():
    """Run all API tests"""
    print("🚀 Starting HaMeR API Tests")
    print("=" * 50)

    base_url = "http://localhost:8000"

    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    max_retries = 30
    for i in range(max_retries):
        if test_health_check(base_url):
            break
        if i < max_retries - 1:
            print(f"   Retrying in 2 seconds... ({i+1}/{max_retries})")
            time.sleep(2)
    else:
        print("❌ Server not ready after 60 seconds")
        return

    print("\n" + "=" * 50)

    # Test directory endpoint
    success_dir = test_directory_endpoint(base_url)

    print("\n" + "=" * 50)

    # Test image list endpoint
    success_img = test_image_list_endpoint(base_url)

    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Health Check: ✅")
    print(f"   Directory Endpoint: {'✅' if success_dir else '❌'}")
    print(f"   Image List Endpoint: {'✅' if success_img else '❌'}")

    if success_dir and success_img:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()