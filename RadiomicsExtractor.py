import os
import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
from Dataload import MedicalImageDataset

def extract_radiomics_features(image, mask):
    """
    Extracts 3D radiomic features from the provided condition image and mask using PyRadiomics.

    Args:
        image (numpy.ndarray): The 3D condition image from which to extract features.
        mask (numpy.ndarray): The corresponding 3D mask for the image.

    Returns:
        dict: A dictionary containing the extracted radiomic features.
    """
    # Convert numpy arrays to SimpleITK images
    image = sitk.GetImageFromArray(image)
    mask = sitk.GetImageFromArray(mask)

    # Ensure the mask is binary
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Initialize the feature extractor
    extractor = featureextractor.RadiomicsFeatureExtractor()

    # Extract features
    features = extractor.execute(image, mask)

    # Filter and return only the original features
    features_dict = {key: features[key] for key in features.keys() if key.startswith('original')}
    return features_dict

def load_radiomics_features(condition_dir, mask_dir):
    """
    Loads and extracts 3D radiomic features for all condition images and corresponding masks.

    Args:
        condition_dir (str): The directory containing the 3D condition images.
        mask_dir (str): The directory containing the corresponding 3D mask images.

    Returns:
        np.ndarray: An array of radiomic features, where each row corresponds to the features of one image.
    """
    dataset = MedicalImageDataset(condition_dir, condition_dir, mask_dir)

    features_list = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        condition_image = sample['condition'].numpy()
        mask_image = sample['mask'].numpy()

        # Extract 3D radiomic features for each condition image and mask pair
        feature_dict = extract_radiomics_features(condition_image, mask_image)
        
        # Convert feature dictionary to a feature vector
        feature_vector = np.array(list(feature_dict.values()))
        features_list.append(feature_vector)

    # Convert list of feature vectors to a 2D numpy array (samples x features)
    features_array = np.vstack(features_list)
    return features_array