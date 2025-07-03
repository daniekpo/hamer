"""
HaMeR Inference Module

This module provides a clean interface for 3D hand reconstruction using HaMeR.
It can be used as a library or by the API server.
"""

import base64
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

import cv2
import numpy as np
import torch

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full

# Import vitpose_model from the project root (not part of the installed package)
import sys
sys.path.append('.')
try:
    from vitpose_model import ViTPoseModel
except ImportError:
    sys.path.append('..')
    from vitpose_model import ViTPoseModel


class HandPrediction:
    """Single hand prediction result"""
    def __init__(self, vertices: List[List[float]], joints: List[List[float]],
                 is_right_hand: bool):
        self.vertices = vertices  # 778x3 vertices
        self.joints = joints      # 21x3 joints
        self.is_right_hand = is_right_hand

    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'vertices': self.vertices,
            'joints': self.joints,
            'is_right_hand': self.is_right_hand,
        }


class HandPredictionResult:
    """Prediction result for a single image"""
    def __init__(self, image_name: Optional[str] = None,
                 hands: Optional[List[HandPrediction]] = None):
        self.image_name = image_name
        self.hands = hands

    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            'image_name': self.image_name,
            'hands': [hand.to_dict() for hand in self.hands] if self.hands else None,
        }


class HamerInference:
    """
    Hamer inference engine for 3D hand reconstruction
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Hamer inference engine

        Args:
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.device = torch.device(device) if device else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )

        print(f"Hamer inference engine initialized on device: {self.device}")
        print("Loading Hamer models...")

        # Download and load Hamer model immediately
        download_models(CACHE_DIR_HAMER)
        self.hamer_model, self.model_cfg = load_hamer(DEFAULT_CHECKPOINT)
        self.hamer_model = self.hamer_model.to(self.device)
        self.hamer_model.eval()

        # Setup body detector (default to vitdet)
        self.detector = self._setup_body_detector('vitdet')
        self._current_detector_type = 'vitdet'

        # Setup keypoint detector
        self.keypoint_detector = ViTPoseModel(self.device)

        print("Hamer models loaded successfully!")



    def _setup_body_detector(self, detector_type: str):
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
            detector = DefaultPredictor_Lazy(detectron2_cfg)

        elif detector_type == 'regnety':
            from detectron2 import model_zoo
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            detector = DefaultPredictor_Lazy(detectron2_cfg)

        else:
            raise ValueError(f"Unknown detector type: {detector_type}")

        # Store detector type for reference
        detector._detector_type = detector_type
        return detector

    def _detect_hands_in_image(self, img_cv2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect hands in an image using body detection and keypoint estimation

        Returns:
            tuple: (bounding_boxes, is_right_flags) for detected hands
        """
        # Detect humans in image
        det_out = self.detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if len(pred_bboxes) == 0:
            return np.array([]), np.array([])

        # Detect human keypoints for each person
        vitposes_out = self.keypoint_detector.predict_pose(
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

    def _predict_hands_from_image(self, img_cv2: np.ndarray, rescale_factor: float = 2.0) -> List[HandPrediction]:
        """
        Predict 3D hand vertices and joints from an image

        Returns:
            List of HandPrediction objects containing vertices, joints, and metadata
        """
        # Detect hands in the image
        boxes, right_flags = self._detect_hands_in_image(img_cv2)

        if len(boxes) == 0:
            return []

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right_flags, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        hand_predictions = []

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                output = self.hamer_model(batch)

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
                    vertices=vertices,
                    joints=joints,
                    is_right_hand=is_right_hand,
                )
                hand_predictions.append(hand_pred)

        # Clear CUDA cache to prevent memory leaks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return hand_predictions

    def _decode_base64_image(self, base64_string: str) -> np.ndarray:
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



    def set_body_detector(self, detector_type: str):
        """
        Set the body detector type

        Args:
            detector_type: Either 'vitdet' or 'regnety'
        """
        if detector_type not in ['vitdet', 'regnety']:
            raise ValueError(f"Unknown detector type: {detector_type}. Must be 'vitdet' or 'regnety'")

        if detector_type != self._current_detector_type:
            self.detector = self._setup_body_detector(detector_type)
            self._current_detector_type = detector_type
            print(f"Body detector set to: {detector_type}")

    def predict(self,
                image_directory: Optional[str] = None,
                images: Optional[Union[List[str], np.ndarray]] = None,
                image_names: Optional[List[str]] = None,
                file_extensions: List[str] = ['*.jpg', '*.png', '*.jpeg'],
                rescale_factor: float = 2.0,
                body_detector: str = 'vitdet') -> List[HandPredictionResult]:
        """
        Main prediction interface

        Args:
            image_directory: Path to directory containing images
            images: List of image paths or base64 encoded strings or numpy arrays
            image_names: Optional list of image names (for base64/array inputs)
            file_extensions: File extensions to process (for directory input)
            rescale_factor: Rescaling factor for processing
            body_detector: Body detector type ('vitdet' or 'regnety')

        Returns:
            Dictionary with results, total_images, successful_predictions, failed_predictions

        Raises:
            ValueError: If both image_directory and images are provided, or if neither is provided
        """
        # Validate input parameters
        if image_directory is not None and images is not None:
            raise ValueError("Cannot provide both image_directory and images. Choose one.")

        if image_directory is None and images is None:
            raise ValueError("Must provide either image_directory or images.")

        # Set body detector if different
        self.set_body_detector(body_detector)

        results = []

        if image_directory is not None:
            results = self._process_directory(image_directory, file_extensions, rescale_factor)
        elif images is not None:
            results = self._process_image_list(images, image_names, rescale_factor)

        return results

    def _process_directory(self, image_directory: str, file_extensions: List[str], rescale_factor: float) -> List[HandPredictionResult]:
        """Process all images in a directory"""
        img_dir = Path(image_directory)
        if not img_dir.exists() or not img_dir.is_dir():
            raise ValueError(f"Directory not found: {image_directory}")

        # Get all image files
        img_paths = []
        for ext in file_extensions:
            img_paths.extend(img_dir.glob(ext))

        if not img_paths:
            raise ValueError(f"No images found in directory with extensions: {file_extensions}")

        results = []
        for img_path in img_paths:
            try:
                img_cv2 = cv2.imread(str(img_path))
                if img_cv2 is None:
                    raise ValueError(f"Failed to load image: {img_path}")

                hands = self._predict_hands_from_image(img_cv2, rescale_factor)
                result =  HandPredictionResult(
                    image_name=img_path.name,
                    hands=hands
                )

                results.append(result)

            except Exception as e:
                continue

        return results

    def _process_image_list(self, images: Union[List[str], np.ndarray], image_names: Optional[List[str]], rescale_factor: float) -> List[HandPredictionResult]:
        """Process a list of images (paths, base64 strings, or numpy arrays)"""
        if image_names and len(image_names) != len(images):
            raise ValueError("Number of image names must match number of images")

        results = []
        for i, image in enumerate(images):
            image_name = image_names[i] if image_names else f"image_{i}"

            try:
                # Handle different input types
                if isinstance(image, str):
                    if os.path.exists(image):
                        # File path
                        img_cv2 = cv2.imread(image)
                        if img_cv2 is None:
                            raise ValueError(f"Failed to load image: {image}")
                    else:
                        # Assume base64 string
                        img_cv2 = self._decode_base64_image(image)
                elif isinstance(image, np.ndarray):
                    # Numpy array
                    img_cv2 = image
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")

                hands = self._predict_hands_from_image(img_cv2, rescale_factor)
                result = HandPredictionResult(
                    image_name=image_name,
                    hands=hands
                )

                results.append(result)

            except Exception as e:
                results.append(HandPredictionResult(
                    image_name=image_name,
                    hands=None
                ))

        return results