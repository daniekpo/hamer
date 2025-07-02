import base64
import io
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from contextlib import asynccontextmanager

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn

# Import the new inference module
from hamer.inference import HaMeRInference, HandPrediction as InferenceHandPrediction, ImagePredictionResult as InferenceImagePredictionResult

# Global inference engine
inference_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup and cleanup on shutdown"""
    global inference_engine

    print("Initializing HaMeR inference engine...")

    # Initialize the inference engine (models will be loaded lazily)
    inference_engine = HaMeRInference()

    print("HaMeR API ready!")

    yield

    # Cleanup (if needed)
    print("Shutting down HaMeR API...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="HaMeR Hand Tracking API",
    description="API for 3D hand reconstruction using HaMeR (Reconstructing Hands in 3D with Transformers)",
    version="1.0.0",
    lifespan=lifespan
)


class HandPrediction(BaseModel):
    """Single hand prediction result"""
    vertices: List[List[float]]  # 778x3 vertices
    joints: List[List[float]]    # 21x3 joints
    is_right_hand: bool
    confidence_score: float


class ImagePredictionResult(BaseModel):
    """Prediction result for a single image"""
    image_path: Optional[str] = None
    image_name: Optional[str] = None
    hands: List[HandPrediction]
    success: bool
    error_message: Optional[str] = None


class DirectoryRequest(BaseModel):
    """Request model for directory-based prediction"""
    image_directory: str
    file_extensions: List[str] = ['*.jpg', '*.png', '*.jpeg']
    rescale_factor: float = 2.0
    body_detector: str = 'vitdet'  # 'vitdet' or 'regnety'


class ImageListRequest(BaseModel):
    """Request model for image list-based prediction"""
    images_base64: List[str]  # List of base64 encoded images
    image_names: Optional[List[str]] = None
    rescale_factor: float = 2.0
    body_detector: str = 'vitdet'


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    results: List[ImagePredictionResult]
    total_images: int
    successful_predictions: int
    failed_predictions: int


def convert_inference_result_to_api_model(inference_result: InferenceImagePredictionResult) -> ImagePredictionResult:
    """Convert inference result to API model"""
    hands = []
    for hand in inference_result.hands:
        hands.append(HandPrediction(
            vertices=hand.vertices,
            joints=hand.joints,
            is_right_hand=hand.is_right_hand,
            confidence_score=hand.confidence_score
        ))

    return ImagePredictionResult(
        image_path=inference_result.image_path,
        image_name=inference_result.image_name,
        hands=hands,
        success=inference_result.success,
        error_message=inference_result.error_message
    )


@app.post("/predict_from_directory", response_model=PredictionResponse)
async def predict_from_directory(request: DirectoryRequest):
    """
    Predict hand poses from all images in a directory

    Args:
        request: DirectoryRequest containing directory path and processing parameters

    Returns:
        PredictionResponse with results for all processed images
    """
    global inference_engine

    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")

    try:
        # Use the inference engine to process the directory
        result = inference_engine.predict(
            image_directory=request.image_directory,
            file_extensions=request.file_extensions,
            rescale_factor=request.rescale_factor,
            body_detector=request.body_detector
        )

        # Convert results to API models
        api_results = []
        for inference_result_dict in result['results']:
            # Convert dict back to inference objects temporarily for conversion
            inference_result = InferenceImagePredictionResult(
                image_path=inference_result_dict['image_path'],
                image_name=inference_result_dict['image_name'],
                hands=[InferenceHandPrediction(**hand_dict) for hand_dict in inference_result_dict['hands']],
                success=inference_result_dict['success'],
                error_message=inference_result_dict['error_message']
            )
            api_results.append(convert_inference_result_to_api_model(inference_result))

        return PredictionResponse(
            results=api_results,
            total_images=result['total_images'],
            successful_predictions=result['successful_predictions'],
            failed_predictions=result['failed_predictions']
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/predict_from_images", response_model=PredictionResponse)
async def predict_from_images(request: ImageListRequest):
    """
    Predict hand poses from a list of base64-encoded images

    Args:
        request: ImageListRequest containing base64 images and processing parameters

    Returns:
        PredictionResponse with results for all processed images
    """
    global inference_engine

    if inference_engine is None:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")

    if not request.images_base64:
        raise HTTPException(status_code=400, detail="No images provided")

    try:
        # Use the inference engine to process the images
        result = inference_engine.predict(
            images=request.images_base64,  # These will be treated as base64 strings
            image_names=request.image_names,
            rescale_factor=request.rescale_factor,
            body_detector=request.body_detector
        )

        # Convert results to API models
        api_results = []
        for inference_result_dict in result['results']:
            # Convert dict back to inference objects temporarily for conversion
            inference_result = InferenceImagePredictionResult(
                image_path=inference_result_dict['image_path'],
                image_name=inference_result_dict['image_name'],
                hands=[InferenceHandPrediction(**hand_dict) for hand_dict in inference_result_dict['hands']],
                success=inference_result_dict['success'],
                error_message=inference_result_dict['error_message']
            )
            api_results.append(convert_inference_result_to_api_model(inference_result))

        return PredictionResponse(
            results=api_results,
            total_images=result['total_images'],
            successful_predictions=result['successful_predictions'],
            failed_predictions=result['failed_predictions']
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global inference_engine

    model_loaded = False
    device_info = "not_initialized"

    if inference_engine is not None:
        model_loaded = inference_engine.hamer_model is not None
        device_info = str(inference_engine.device)

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": device_info
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "HaMeR Hand Tracking API",
        "description": "API for 3D hand reconstruction using HaMeR (Reconstructing Hands in 3D with Transformers)",
        "endpoints": {
            "/predict_from_directory": "Predict from image directory",
            "/predict_from_images": "Predict from base64 image list",
            "/health": "Health check",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)