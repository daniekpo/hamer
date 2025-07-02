#!/usr/bin/env python3
"""
Test script for HaMeR Library Interface

This script demonstrates how to use the HaMeR inference library directly
without needing to run the API server.
"""

import sys
from pathlib import Path

# Import the HaMeR inference library
from hamer.inference import predict, HaMeRInference


def test_directory_inference():
    """Test library interface with directory"""
    print("🔬 Testing HaMeR library with directory...")

    image_dir = "example_data"

    # Check if directory exists
    if not Path(image_dir).exists():
        print(f"❌ Directory not found: {image_dir}")
        return False

    try:
        # Use the simple predict function
        result = predict(
            image_directory=image_dir,
            file_extensions=['*.jpg', '*.png', '*.jpeg'],
            rescale_factor=2.0,
            body_detector='vitdet'
        )

        print(f"✅ Directory processing successful!")
        print(f"   Total images: {result['total_images']}")
        print(f"   Successful: {result['successful_predictions']}")
        print(f"   Failed: {result['failed_predictions']}")

        # Print details for first few results
        for i, img_result in enumerate(result["results"][:3]):
            print(f"   Image {i + 1}: {img_result['image_name']}")
            print(f"     Hands detected: {len(img_result['hands'])}")

            for j, hand in enumerate(img_result["hands"]):
                hand_type = "Right" if hand["is_right_hand"] else "Left"
                print(f"       Hand {j + 1}: {hand_type} hand")
                print(f"         Vertices: {len(hand['vertices'])} points")
                print(f"         Joints: {len(hand['joints'])} points")
                print(f"         Confidence: {hand['confidence_score']}")

        return True

    except Exception as e:
        print(f"❌ Directory processing failed: {e}")
        return False


def test_image_list_inference():
    """Test library interface with image list"""
    print("🖼️ Testing HaMeR library with image list...")

    test_image_path = "example_data/test1.jpg"

    # Check if test image exists
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False

    try:
        # Use the simple predict function with image paths
        result = predict(
            images=[test_image_path],  # List of image paths
            image_names=["test_image.jpg"],
            rescale_factor=2.0,
            body_detector='vitdet'
        )

        print(f"✅ Image list processing successful!")
        print(f"   Total images: {result['total_images']}")
        print(f"   Successful: {result['successful_predictions']}")
        print(f"   Failed: {result['failed_predictions']}")

        # Print details for the result
        if result["results"]:
            img_result = result["results"][0]
            print(f"   Image: {img_result['image_name']}")
            print(f"     Hands detected: {len(img_result['hands'])}")

            for j, hand in enumerate(img_result["hands"]):
                hand_type = "Right" if hand["is_right_hand"] else "Left"
                print(f"       Hand {j + 1}: {hand_type} hand")
                print(f"         Vertices: {len(hand['vertices'])} points")
                print(f"         Joints: {len(hand['joints'])} points")
                print(f"         Confidence: {hand['confidence_score']}")

        return True

    except Exception as e:
        print(f"❌ Image list processing failed: {e}")
        return False


def test_inference_engine_class():
    """Test using the HaMeRInference class directly"""
    print("⚙️ Testing HaMeRInference class directly...")

    test_image_path = "example_data/test1.jpg"

    # Check if test image exists
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False

    try:
        # Create inference engine
        engine = HaMeRInference(device='cuda' if True else 'cpu')  # Auto-detect device

        # Test with image list
        result = engine.predict(
            images=[test_image_path],
            image_names=["direct_test.jpg"],
            rescale_factor=2.0,
            body_detector='vitdet'
        )

        print(f"✅ Direct inference engine test successful!")
        print(f"   Total images: {result['total_images']}")
        print(f"   Successful: {result['successful_predictions']}")
        print(f"   Failed: {result['failed_predictions']}")

        return True

    except Exception as e:
        print(f"❌ Direct inference engine test failed: {e}")
        return False


def test_input_validation():
    """Test input validation"""
    print("🛡️ Testing input validation...")

    try:
        # Test providing both directory and images (should fail)
        result = predict(
            image_directory="example_data",
            images=["test.jpg"],
        )
        print("❌ Should have failed with both inputs provided")
        return False

    except ValueError as e:
        print(f"✅ Correctly caught validation error: {e}")

    try:
        # Test providing neither directory nor images (should fail)
        result = predict()
        print("❌ Should have failed with no inputs provided")
        return False

    except ValueError as e:
        print(f"✅ Correctly caught validation error: {e}")

    return True


def main():
    """Run all library tests"""
    print("🚀 Starting HaMeR Library Tests")
    print("=" * 50)

    # Test input validation first
    success_validation = test_input_validation()

    print("\n" + "=" * 50)

    # Test directory processing
    success_dir = test_directory_inference()

    print("\n" + "=" * 50)

    # Test image list processing
    success_img = test_image_list_inference()

    print("\n" + "=" * 50)

    # Test direct inference engine usage
    success_engine = test_inference_engine_class()

    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    print(f"   Input Validation: {'✅' if success_validation else '❌'}")
    print(f"   Directory Processing: {'✅' if success_dir else '❌'}")
    print(f"   Image List Processing: {'✅' if success_img else '❌'}")
    print(f"   Direct Engine Usage: {'✅' if success_engine else '❌'}")

    if all([success_validation, success_dir, success_img, success_engine]):
        print("\n🎉 All library tests passed!")
        print("\n📝 Usage Examples:")
        print("   # Simple directory processing:")
        print("   from hamer.inference import predict")
        print("   result = predict(image_directory='path/to/images')")
        print("")
        print("   # Process list of image paths:")
        print("   result = predict(images=['img1.jpg', 'img2.jpg'])")
        print("")
        print("   # Using inference engine directly:")
        print("   from hamer.inference import HaMeRInference")
        print("   engine = HaMeRInference()")
        print("   result = engine.predict(image_directory='path/to/images')")
    else:
        print("\n⚠️ Some library tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()