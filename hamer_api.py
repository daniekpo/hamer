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

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full

# Import vitpose_model from the hamer directory (not part of the installed package)
import sys
sys.path.append('./hamer')
from vitpose_model import ViTPoseModel

# Global variables to store the models
hamer_model = None
model_cfg = None
detector = None
keypoint_detector = None
device = None


def setup_body_detector(detector_type: str):
    """Setup the body detector (vitdet or regnety)"""
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    if detector_type == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        return DefaultPredictor_Lazy(detectron2_cfg)

    elif detector_type == 'regnety':
        from detectron2 import model_zoo
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        return DefaultPredictor_Lazy(detectron2_cfg)

    else:
        raise ValueError(f"Unknown detector type: {detector_type}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup and cleanup on shutdown"""
    global hamer_model, model_cfg, detector, keypoint_detector, device

    print("Loading HaMeR models...")

    # Setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Download and load HaMeR model
    download_models(CACHE_DIR_HAMER)
    hamer_model, model_cfg = load_hamer(DEFAULT_CHECKPOINT)
    hamer_model = hamer_model.to(device)
    hamer_model.eval()

    # Setup body detector (default to vitdet)
    detector = setup_body_detector('vitdet')

    # Setup keypoint detector
    keypoint_detector = ViTPoseModel(device)

    print("HaMeR models loaded successfully!")

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


def detect_hands_in_image(img_cv2: np.ndarray, detector, keypoint_detector) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect hands in an image using body detection and keypoint estimation

    Returns:
        tuple: (bounding_boxes, is_right_flags) for detected hands
    """
    # Detect humans in image
    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    if len(pred_bboxes) == 0:
        return np.array([]), np.array([])

    # Detect human keypoints for each person
    vitposes_out = keypoint_detector.predict_pose(
        img,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []

    # Extract hand bounding boxes from keypoint detections
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes['keypoints'][-42:-21]
        right_hand_keyp = vitposes['keypoints'][-21:]

        # Process left hand
        keyp = left_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                   keyp[valid, 0].max(), keyp[valid, 1].max()]
            bboxes.append(bbox)
            is_right.append(0)

        # Process right hand
        keyp = right_hand_keyp
        valid = keyp[:, 2] > 0.5
        if sum(valid) > 3:
            bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(),
                   keyp[valid, 0].max(), keyp[valid, 1].max()]
            bboxes.append(bbox)
            is_right.append(1)

    if len(bboxes) == 0:
        return np.array([]), np.array([])

    return np.stack(bboxes), np.stack(is_right)


def predict_hands_from_image(img_cv2: np.ndarray, rescale_factor: float = 2.0) -> List[HandPrediction]:
    """
    Predict 3D hand vertices and joints from an image

    Returns:
        List of HandPrediction objects containing vertices, joints, and metadata
    """
    global hamer_model, model_cfg, detector, keypoint_detector, device

    # Detect hands in the image
    boxes, right_flags = detect_hands_in_image(img_cv2, detector, keypoint_detector)

    if len(boxes) == 0:
        return []

    # Run reconstruction on all detected hands
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right_flags, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    hand_predictions = []

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            output = hamer_model(batch)

        # Extract predictions
        batch_size = batch['img'].shape[0]

        for n in range(batch_size):
            # Get vertices (778x3)
            vertices = output['pred_vertices'][n].detach().cpu().numpy()

            # Get joints (21x3)
            joints = output['pred_keypoints_3d'][n].detach().cpu().numpy()

            # Get hand orientation info
            is_right_hand = bool(batch['right'][n].cpu().numpy())

            # Apply hand orientation correction to vertices
            if not is_right_hand:
                vertices[:, 0] = -vertices[:, 0]

            # Create prediction object
            hand_pred = HandPrediction(
                vertices=vertices.tolist(),
                joints=joints.tolist(),
                is_right_hand=is_right_hand,
                confidence_score=1.0  # Could be improved with actual confidence scores
            )
            hand_predictions.append(hand_pred)

    # Clear CUDA cache to prevent memory leaks
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return hand_predictions


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image"""
    try:
        img_data = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")
        return img
    except Exception as e:
        raise ValueError(f"Invalid base64 image data: {str(e)}")


def process_single_image(img_cv2: np.ndarray, image_name: str, rescale_factor: float) -> ImagePredictionResult:
    """Process a single image and return prediction results"""
    try:
        hands = predict_hands_from_image(img_cv2, rescale_factor)
        return ImagePredictionResult(
            image_name=image_name,
            hands=hands,
            success=True
        )
    except Exception as e:
        return ImagePredictionResult(
            image_name=image_name,
            hands=[],
            success=False,
            error_message=str(e)
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
    global detector

    # Validate directory
    img_dir = Path(request.image_directory)
    if not img_dir.exists() or not img_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.image_directory}")

    # Setup detector if different from current
    current_detector_type = getattr(detector, '_detector_type', 'vitdet')
    if request.body_detector != current_detector_type:
        detector = setup_body_detector(request.body_detector)
        detector._detector_type = request.body_detector

    # Get all image files
    img_paths = []
    for ext in request.file_extensions:
        img_paths.extend(img_dir.glob(ext))

    if not img_paths:
        raise HTTPException(status_code=400, detail=f"No images found in directory with extensions: {request.file_extensions}")

    results = []
    successful = 0
    failed = 0

    # Process each image
    for img_path in img_paths:
        try:
            img_cv2 = cv2.imread(str(img_path))
            if img_cv2 is None:
                raise ValueError(f"Failed to load image: {img_path}")

            result = process_single_image(img_cv2, img_path.name, request.rescale_factor)
            result.image_path = str(img_path)
            results.append(result)

            if result.success:
                successful += 1
            else:
                failed += 1

        except Exception as e:
            results.append(ImagePredictionResult(
                image_path=str(img_path),
                image_name=img_path.name,
                hands=[],
                success=False,
                error_message=str(e)
            ))
            failed += 1

    return PredictionResponse(
        results=results,
        total_images=len(img_paths),
        successful_predictions=successful,
        failed_predictions=failed
    )


@app.post("/predict_from_images", response_model=PredictionResponse)
async def predict_from_images(request: ImageListRequest):
    """
    Predict hand poses from a list of base64-encoded images

    Args:
        request: ImageListRequest containing base64 images and processing parameters

    Returns:
        PredictionResponse with results for all processed images
    """
    global detector

    if not request.images_base64:
        raise HTTPException(status_code=400, detail="No images provided")

    # Setup detector if different from current
    current_detector_type = getattr(detector, '_detector_type', 'vitdet')
    if request.body_detector != current_detector_type:
        detector = setup_body_detector(request.body_detector)
        detector._detector_type = request.body_detector

    # Validate image names if provided
    if request.image_names and len(request.image_names) != len(request.images_base64):
        raise HTTPException(status_code=400, detail="Number of image names must match number of images")

    results = []
    successful = 0
    failed = 0

    # Process each image
    for i, base64_img in enumerate(request.images_base64):
        image_name = request.image_names[i] if request.image_names else f"image_{i}.jpg"

        try:
            img_cv2 = decode_base64_image(base64_img)
            result = process_single_image(img_cv2, image_name, request.rescale_factor)
            results.append(result)

            if result.success:
                successful += 1
            else:
                failed += 1

        except Exception as e:
            results.append(ImagePredictionResult(
                image_name=image_name,
                hands=[],
                success=False,
                error_message=str(e)
            ))
            failed += 1

    return PredictionResponse(
        results=results,
        total_images=len(request.images_base64),
        successful_predictions=successful,
        failed_predictions=failed
    )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": hamer_model is not None,
        "device": str(device) if device else "not_set"
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