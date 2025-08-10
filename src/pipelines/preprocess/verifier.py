import os
import json
import logging
import zipfile
import tempfile
import shutil
from .config import SUPPORTED_AUGMENTATIONS
from ..load.config import SUPPORTED_DATASETS

logger = logging.getLogger("AugmentationPipeline")

# Optional HF import with fallback
try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("huggingface_hub not available. Dataset auto-download disabled.")

IMAGE_FORMAT = "png"

# Hugging Face dataset configuration
HF_USERNAME = "ianisdev"  # Your HF username


def count_images_in_dir(directory, ext=IMAGE_FORMAT):
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) if f.endswith(f".{ext}")])

def load_metadata(path):
    meta_path = os.path.join(path, "meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        return json.load(f)

def verify_preprocessed_split(
    dataset_name: str,
    augmentation: str = None,
    split: str = "train",
    expected_count: int = None,
    root: str = "./processed",
    image_format: str = IMAGE_FORMAT,
):
    if split == "train" and not augmentation:
        raise ValueError("Augmentation name must be provided for train split.")

    # Build correct path
    if split == "train":
        split_dir = os.path.join(root, dataset_name, "train", augmentation)
    else:
        split_dir = os.path.join(root, dataset_name, "test")

    if not os.path.exists(split_dir):
        logger.warning(f"⛔ Preprocessed split not found: {split_dir}")
        return False

    img_count = count_images_in_dir(split_dir, ext=image_format)
    metadata = load_metadata(split_dir)

    if expected_count is not None and img_count != expected_count:
        logger.warning(f"⚠️ Image count mismatch in {split_dir}: expected {expected_count}, found {img_count}")
        return False

    if metadata:
        logger.info(f"✅ Metadata loaded: {split_dir}/meta.json")
        logger.debug(f"  └── {metadata}")
    else:
        logger.warning(f"ℹ️ Metadata not found in {split_dir}, relying on file count only")

    logger.info(f"✅ Verified {dataset_name}-{split}-{augmentation or 'none'} ({img_count} images)")
    return True


def verify_all_preprocessed(
    dataset: str,
    augmentation: str,
    root: str = "./processed",
    image_format: str = IMAGE_FORMAT,
):
    datasets_to_check = SUPPORTED_DATASETS if dataset == "all" else [dataset]
    augmentations_to_check = SUPPORTED_AUGMENTATIONS if augmentation == "all" else [augmentation]

    for ds in datasets_to_check:
        if ds == "cifar":
            ds = "cifar10"
        # Always verify test set
        if not verify_preprocessed_split(ds, split="test", root=root, image_format=image_format):
            raise FileNotFoundError(f"Test set missing or corrupted for dataset: {ds}")

        # Now verify all augmentations for train
        for aug in augmentations_to_check:
            if not verify_preprocessed_split(ds, aug, split="train", root=root, image_format=image_format):
                raise FileNotFoundError(f"Train set missing for dataset '{ds}' and augmentation '{aug}'")


def verify_datasets_ready_for_training(
    dataset: str,
    augmentation: str,
    root: str = "./processed",
    image_format: str = IMAGE_FORMAT,
):
    """
    Verifies that processed datasets are ready for training.
    Checks both test and train datasets with proper augmentation support.
    Called before training starts to ensure all required data is available.
    """
    logger.info("🔍 Verifying processed datasets are ready for training...")
    
    datasets_to_check = SUPPORTED_DATASETS if dataset == "all" else [dataset]
    augmentations_to_check = SUPPORTED_AUGMENTATIONS if augmentation == "all" else [augmentation]
    
    missing_datasets = []
    empty_datasets = []
    
    for ds in datasets_to_check:
        if ds == "cifar":
            ds = "cifar10"
            
        logger.info(f"📊 Checking dataset: {ds}")
        
        # Check if dataset directory exists
        dataset_path = os.path.join(root, ds)
        if not os.path.exists(dataset_path):
            missing_datasets.append(ds)
            logger.error(f"❌ Dataset directory missing: {dataset_path}")
            continue
            
        # Check test set (no augmentation)
        test_path = os.path.join(dataset_path, "test")
        if not os.path.exists(test_path):
            missing_datasets.append(f"{ds}/test")
            logger.error(f"❌ Test directory missing: {test_path}")
        else:
            test_count = count_images_in_dir(test_path, image_format)
            if test_count == 0:
                empty_datasets.append(f"{ds}/test")
                logger.error(f"❌ Test set is empty: {test_path}")
            else:
                logger.info(f"✅ Test set verified: {test_count} images in {ds}/test")
        
        # Check train sets with augmentations
        for aug in augmentations_to_check:
            if aug == "traditional":
                # Traditional augmentation uses base train directory
                train_path = os.path.join(dataset_path, "train")
            else:
                # Other augmentations have their own subdirectories
                train_path = os.path.join(dataset_path, "train", aug)
                
            if not os.path.exists(train_path):
                missing_datasets.append(f"{ds}/train/{aug}")
                logger.error(f"❌ Train directory missing: {train_path}")
            else:
                train_count = count_images_in_dir(train_path, image_format)
                if train_count == 0:
                    empty_datasets.append(f"{ds}/train/{aug}")
                    logger.error(f"❌ Train set is empty: {train_path}")
                else:
                    logger.info(f"✅ Train set verified: {train_count} images in {ds}/train/{aug}")
    
    # Report any issues
    if missing_datasets:
        raise FileNotFoundError(
            f"Missing processed datasets: {missing_datasets}. "
            f"Please run preprocessing with --preprocess flag first."
        )
        
    if empty_datasets:
        raise ValueError(
            f"Empty processed datasets: {empty_datasets}. "
            f"Please re-run preprocessing to generate data."
        )
    
    logger.info("🎉 All datasets verified and ready for training!")


def download_augmented_trainset_from_hf(dataset_name, augmentation="augmented", root="./processed", max_retries=3):
    """
    Download complete augmented dataset from Hugging Face Hub and extract it
    This includes both test and train data with all augmentation types
    """
    if not HF_AVAILABLE:
        logger.error("❌ huggingface_hub not available. Please install with: pip install huggingface_hub")
        return False
    
    # The repo is named {dataset}_augmented but contains {dataset}.zip
    repo_name = f"{dataset_name}_augmented"
    repo_id = f"{HF_USERNAME}/{repo_name}"
    zip_filename = f"{dataset_name}.zip"  # Expected filename
    
    logger.info(f"📥 Downloading {dataset_name} complete dataset from HF Hub...")
    logger.info(f"📂 Repository: {repo_id}")
    logger.info(f"📄 File: {zip_filename}")
    
    # Special handling for ImageNet (larger dataset)
    if dataset_name == "imagenet":
        logger.info("⚠️  ImageNet is a large dataset - download may take 10+ minutes")
        max_retries = 2  # Fewer retries for large dataset
    
    # First, check what files are actually available in the repository
    try:
        from huggingface_hub import list_repo_files
        available_files = list_repo_files(repo_id, repo_type="dataset")
        zip_files = [f for f in available_files if f.endswith('.zip')]
        
        if zip_filename not in available_files:
            if zip_files:
                # Use the first available zip file
                actual_filename = zip_files[0]
                logger.warning(f"⚠️  Expected file {zip_filename} not found, using {actual_filename}")
                zip_filename = actual_filename
            else:
                logger.error(f"❌ No ZIP files found in repository {repo_id}")
                logger.info(f"💡 Available files: {available_files[:5]}")
                return False
        else:
            logger.info(f"✅ Found expected file: {zip_filename}")
            
    except Exception as e:
        logger.warning(f"⚠️  Could not list repository files: {e}")
        logger.info("💡 Proceeding with expected filename...")
    
    for attempt in range(max_retries):
        logger.info(f"📥 Attempt {attempt + 1}/{max_retries}: Starting download...")
        try:
            # Download ZIP file with fresh cache directory each time
            cache_dir = tempfile.mkdtemp()
            zip_path = hf_hub_download(
                repo_id=repo_id,
                filename=zip_filename,
                repo_type="dataset",
                cache_dir=cache_dir,
                force_download=True  # Force fresh download on retries
            )
            
            # Check file size
            zip_size = os.path.getsize(zip_path)
            zip_size_gb = zip_size / (1024**3)
            logger.info(f"✅ Downloaded {zip_filename} ({zip_size_gb:.2f} GB)")
            
            # Create target directory structure
            dataset_dir = os.path.join(root, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Extract ZIP file to a temporary directory first
            temp_extract_dir = tempfile.mkdtemp()
            logger.info(f"📦 Extracting {zip_filename}...")
            
            # Extract with filtering and error tolerance
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                extracted_files = 0
                corrupted_files = 0
                
                for file_info in zip_ref.filelist:
                    # Skip macOS metadata files
                    if file_info.filename.startswith('__MACOSX/') or '._' in file_info.filename:
                        logger.debug(f"Skipping macOS metadata file: {file_info.filename}")
                        continue
                    
                    try:
                        # Extract individual file
                        zip_ref.extract(file_info, temp_extract_dir)
                        extracted_files += 1
                    except (zipfile.BadZipFile, Exception) as e:
                        logger.warning(f"⚠️  Skipping corrupted file: {file_info.filename} ({e})")
                        corrupted_files += 1
                        continue
                
                logger.info(f"📊 Extraction complete: {extracted_files} files extracted, {corrupted_files} files skipped")
                
                # If too many files are corrupted, consider it a failed attempt
                if corrupted_files > extracted_files * 0.1:  # More than 10% corrupted
                    raise zipfile.BadZipFile(f"Too many corrupted files: {corrupted_files}/{extracted_files + corrupted_files}")
            
            # The ZIP contains the full dataset structure: {dataset}/test/ and {dataset}/train/
            # Move the extracted content to the proper location
            extracted_dataset_dir = os.path.join(temp_extract_dir, dataset_name)
            
            if os.path.exists(extracted_dataset_dir):
                # Move test directory
                extracted_test_dir = os.path.join(extracted_dataset_dir, "test")
                target_test_dir = os.path.join(dataset_dir, "test")
                if os.path.exists(extracted_test_dir):
                    if os.path.exists(target_test_dir):
                        shutil.rmtree(target_test_dir)
                    shutil.move(extracted_test_dir, target_test_dir)
                    test_count = count_images_in_dir(target_test_dir, IMAGE_FORMAT)
                    logger.info(f"✅ Extracted {test_count} test images to {target_test_dir}")
                
                # Move train directory
                extracted_train_dir = os.path.join(extracted_dataset_dir, "train")
                target_train_dir = os.path.join(dataset_dir, "train")
                if os.path.exists(extracted_train_dir):
                    if os.path.exists(target_train_dir):
                        shutil.rmtree(target_train_dir)
                    shutil.move(extracted_train_dir, target_train_dir)
                    train_count = count_images_in_dir(target_train_dir, IMAGE_FORMAT)
                    logger.info(f"✅ Extracted {train_count} training images to {target_train_dir}")
            else:
                raise FileNotFoundError(f"Expected directory {dataset_name} not found in ZIP")
            
            # Clean up temporary files
            shutil.rmtree(temp_extract_dir)
            shutil.rmtree(cache_dir)
            
            logger.info(f"🎉 Successfully downloaded and extracted complete {dataset_name} dataset!")
            return True
            
        except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
            logger.warning(f"⚠️  ZIP extraction issues on attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                logger.info(f"🔄 Retrying download...")
                # Clean up failed attempt
                try:
                    if 'temp_extract_dir' in locals():
                        shutil.rmtree(temp_extract_dir)
                    if 'cache_dir' in locals():
                        shutil.rmtree(cache_dir)
                except:
                    pass
            else:
                logger.error(f"❌ All {max_retries} download attempts had extraction issues")
                # Try to check if we got at least some useful data
                try:
                    if 'temp_extract_dir' in locals() and os.path.exists(temp_extract_dir):
                        test_files = 0
                        train_files = 0
                        for root_dir, dirs, files in os.walk(temp_extract_dir):
                            if '/test/' in root_dir:
                                test_files += len([f for f in files if f.endswith('.png')])
                            elif '/train/' in root_dir:
                                train_files += len([f for f in files if f.endswith('.png')])
                        
                        if test_files > 100 and train_files > 1000:  # Minimum threshold
                            logger.warning(f"⚠️  Proceeding with partial dataset: {test_files} test, {train_files} train images")
                            # Continue with partial extraction
                        else:
                            logger.error(f"❌ Insufficient data extracted: {test_files} test, {train_files} train images")
                            return False
                    else:
                        return False
                except:
                    return False
        except Exception as e:
            logger.error(f"❌ Failed to download {dataset_name} dataset from HF: {e}")
            logger.info(f"💡 Tried repository: {repo_id}")
            # Clean up on other errors
            try:
                if 'temp_extract_dir' in locals():
                    shutil.rmtree(temp_extract_dir)
                if 'cache_dir' in locals():
                    shutil.rmtree(cache_dir)
            except:
                pass
            return False
    
    return False


def auto_setup_missing_datasets(
    dataset: str,
    augmentation: str,
    root: str = "./processed",
):
    """
    Automatically download and set up missing datasets from Hugging Face
    Downloads augmented datasets which contain both test and train data
    """
    logger.info("🔍 Checking for missing datasets and attempting auto-download...")
    
    datasets_to_check = SUPPORTED_DATASETS if dataset == "all" else [dataset]
    downloaded_any = False
    
    for ds in datasets_to_check:
        if ds == "cifar":
            ds = "cifar10"
            
        dataset_path = os.path.join(root, ds)
        test_path = os.path.join(dataset_path, "test")
        train_path = os.path.join(dataset_path, "train")
        
        logger.info(f"📊 Checking dataset: {ds}")
        
        # Check if we need to download the complete dataset
        needs_download = False
        
        if not os.path.exists(test_path) or count_images_in_dir(test_path, IMAGE_FORMAT) == 0:
            logger.info(f"❌ Test dataset missing for {ds}")
            needs_download = True
        
        # For train, check if there are any augmentation subdirectories with images
        train_has_data = False
        if os.path.exists(train_path):
            # Check for direct images in train/
            if count_images_in_dir(train_path, IMAGE_FORMAT) > 0:
                train_has_data = True
            else:
                # Check for images in augmentation subdirectories
                for item in os.listdir(train_path):
                    subdir_path = os.path.join(train_path, item)
                    if os.path.isdir(subdir_path) and count_images_in_dir(subdir_path, IMAGE_FORMAT) > 0:
                        train_has_data = True
                        break
        
        if not train_has_data:
            logger.info(f"❌ Training dataset missing for {ds}")
            needs_download = True
        
        if needs_download:
            logger.info(f"📥 Downloading complete {ds} dataset (test + train) from HF Hub...")
            
            if download_augmented_trainset_from_hf(ds, "all", root):
                downloaded_any = True
                logger.info(f"✅ Successfully downloaded complete {ds} dataset")
            else:
                logger.warning(f"⚠️  Could not auto-download {ds} dataset")
                logger.info(f"💡 Alternative: Run preprocessing to generate data locally:")
                logger.info(f"💡   python src/main.py --preprocess --dataset {ds} --augment all")
        else:
            logger.info(f"✅ Complete dataset found for {ds}")
    
    if downloaded_any:
        logger.info("🎉 Successfully downloaded missing datasets!")
        logger.info("📦 Downloaded datasets are ready for training!")
    else:
        logger.info("ℹ️  All datasets already available locally")
    
    return downloaded_any


def smart_verify_datasets_ready_for_training(
    dataset: str,
    augmentation: str,
    root: str = "./processed",
    auto_download: bool = True,
):
    """
    Enhanced verifier that can auto-download missing datasets
    Downloads complete augmented datasets which include both test and train data
    """
    logger.info("🔍 Smart verification: checking datasets and auto-downloading if needed...")
    
    # First, try auto-download missing datasets
    if auto_download:
        auto_setup_missing_datasets(dataset, augmentation, root)
    
    # Then run normal verification
    try:
        verify_datasets_ready_for_training(dataset, augmentation, root)
        logger.info("🎉 All datasets verified and ready for training!")
        return True
    except FileNotFoundError as e:
        logger.error(f"❌ Dataset verification failed: {e}")
        logger.info("💡 Tip: Complete datasets can be auto-downloaded from HF Hub")
        logger.info(f"💡 Available: ianisdev/{{dataset}}_augmented (contains both test and train)")
        logger.info("💡 Alternative: Run preprocessing with --preprocess flag")
        return False
    except Exception as e:
        logger.error(f"❌ Verification error: {e}")
        return False
