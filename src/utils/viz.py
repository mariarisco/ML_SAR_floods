import os
import rasterio as rio
import data_preprocessing as dp

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_features(image_index, image_dir,label_dir):
    """
    Load one image, compute features, and visualize each feature as an image.
    
    Parameters:
    - image_index (int): Index of the image (e.g., 0 for 0.tif)
    - image_dir (str): Path to directory containing .tif images
    """
    
    img_path = os.path.join(image_dir, f"{image_index}.tif")
    lbl_path = os.path.join(label_dir, f"{image_index}.png")
    
    # Load bands and DEM
    bands, dem = dp.load_image_bands(img_path)
    b1, b2, b3, b4, b5, b6 = bands
    
    # Compute features
    rat_vh_vv, rat_vv_vh, rat_norm = dp.compute_sar_ratios(b1, b2)
    slope = dp.compute_slope_img(dem)
    aspect = dp.compute_aspect_img(dem)
    
    # Load label mask
    with rio.open(lbl_path) as lbl_src:
        label_mask = lbl_src.read(1).astype('float32')

    features = {
        'VV': b1,
        'VH': b2,
        'VH/VV': rat_vh_vv,
        'VV/VH': rat_vv_vh,
        'NORM': rat_norm,
        'DEM_mer': b3,
        'DEM_cop': b4,
        'SLOPE': slope,
        'ASPECT': aspect,
        'WCM': b5,
        'WOP': b6,
        'Label (Water Mask)': label_mask
    }

    # Plot all features
    n_features = len(features)
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(4 * n_cols, 4 * n_rows))
    
    for i, (name, data) in enumerate(features.items(), start=1):
        plt.subplot(n_rows, n_cols, i)
        plt.imshow(data, cmap='gray')
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_feature_histograms(df, bins=50):
    num_cols = df.shape[1]
    n_cols = 3  # number of subplots per row
    n_rows = (num_cols + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 5, n_rows * 4))

    for i, col in enumerate(df.columns, start=1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], bins=bins, kde=False, color='steelblue')
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

