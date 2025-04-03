import os
from tqdm.notebook import tqdm
import time
import random

import numpy as np
import pandas as pd
import rasterio as rio
import richdem as rd
import matplotlib.pyplot as plt

# --- Slope & Aspect with caching ---
def compute_slope(fname, dem_data, cache_dir='slope_cache'):
    # os.makedirs(cache_dir, exist_ok=True)
    # path = f'{cache_dir}/{i}.npy'
    # if os.path.exists(path):
    #     return np.load(path)
    sp_dem = rd.rdarray(dem_data, no_data=-9999)
    slope = rd.TerrainAttribute(sp_dem, attrib='slope_percentage')
    # np.save(path, slope)
    return slope

def compute_aspect(fname, dem_data, cache_dir='aspect_cache'):
    # os.makedirs(cache_dir, exist_ok=True)
    # path = f'{cache_dir}/{i}.npy'
    # if os.path.exists(path):
    #     return np.load(path)
    sp_dem = rd.rdarray(dem_data, no_data=-9999)
    aspect = rd.TerrainAttribute(sp_dem, attrib='aspect')
    # np.save(path, aspect)
    return aspect

def compute_slope_img(dem_data):
    sp_dem = rd.rdarray(dem_data, no_data=-9999)
    return rd.TerrainAttribute(sp_dem, attrib='slope_percentage')

def compute_aspect_img(dem_data):
    sp_dem = rd.rdarray(dem_data, no_data=-9999)
    return rd.TerrainAttribute(sp_dem, attrib='aspect')

def compute_sar_ratios(vv, vh):
    rat_vh_vv = vh / vv
    rat_vv_vh = vv / vh
    rat_norm = (vh - vv) / (vh + vv)
    return rat_vh_vv, rat_vv_vh, rat_norm

def load_image_bands(image_path):
    with rio.open(image_path) as src:
        bands = [src.read(i).astype('float32') for i in range(1, 7)] # float 32 to avoid RAM memory problems. Change to float64 in case of more coputational capacity
        dem = bands[4]  # DEM to be used for slope and aspect (DEM Copernicus)
    return bands, dem

def flatten_and_stack(features_dict):
    flat_data = {}
    for key, arr in features_dict.items():
        flat_data[key] = arr.flatten()
    return flat_data

def process_images(M, image_dir, label_dir):
    # file_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    # n = int(M * len(file_list))

    # Get all .tif filenames and shuffle them
    file_list = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])
    total_images = len(file_list)
    n = int(M * total_images)
    
    random.seed(43)
    selected_files = random.sample(file_list, n)

    all_features = []
    all_labels = []

    for fname in tqdm(selected_files, desc='Image processing'):
    # for  in tqdm(range(n), desc='Procesando imágenes'):
        total_start = time.time()

        # Get base index (strip .tif for label path)
        base_name = os.path.splitext(fname)[0]

        # ---- Load image bands and DEM ----
        t0 = time.time()
        img_path = os.path.join(image_dir, fname)
        lbl_path = os.path.join(label_dir, f"{base_name}.png")
        # img_path = os.path.join(image_dir, f"{i}.tif")
        # lbl_path = os.path.join(label_dir, f"{i}.png")
        bands, dem = load_image_bands(img_path)
        b1, b2, b3, b4, b5, b6 = bands
        t1 = time.time()

        # ---- SAR features calculation ----
        t2 = time.time()
        rat_vh_vv, rat_vv_vh, rat_norm = compute_sar_ratios(b1, b2)
        t3 = time.time()

        # ---- Terrain features calculation ----
        t4 = time.time()
        slope = compute_slope(fname, dem)        
        aspect = compute_aspect(fname, dem)
        t5 = time.time()

        # ---- Label data loading ----
        t6 = time.time()
        label_data = rio.open(lbl_path).read(1).astype('float32')
        t7 = time.time()

        # ---- Flatten & stack features ----
        t8 = time.time()
        feature_stack = {
            'VV': b1,
            'VH': b2,
            'VH_VV': rat_vh_vv,
            'VV_VH': rat_vv_vh,
            'NORM': rat_norm,
            'DEM_mer': b3,
            'DEM_cop': b4,
            'SLOPE': slope,
            'ASPECT': aspect,
            'WCM': b5,
            'WOP': b6
        }
        flat_feats = flatten_and_stack(feature_stack)
        df_feats = pd.DataFrame(flat_feats).astype('float32')
        df_feats['Label'] = label_data.flatten().astype('float32')
        t9 = time.time()

        # ---- Append to lists ----
        all_features.append(df_feats.drop(columns='Label'))
        all_labels.append(df_feats['Label'])

        total_end = time.time()

        print(
            f"[Image {fname}] Total: {total_end - total_start:.2f}s | "
            f"Load: {t1 - t0:.2f}s | SAR: {t3 - t2:.2f}s | "
            f"Slope & Aspect: {t5 - t4:.2f}s"
            f"Label: {t7 - t6:.2f}s | Flatten & Stack: {t9 - t8:.2f}s"
        )


    X = pd.concat(all_features, ignore_index=True)
    y = pd.concat(all_labels, ignore_index=True)
    y.name ='Labels'

    print(f"Nº of images processed: {int(n)}")

    return X, y


def bin_aspect_to_direction(df, column='ASPECT'):
    # First, create a copy of the aspect column
    aspect = df[column] % 360

    # Create a new column initialized with NaN
    df['ORIENT'] = pd.Series(index=df.index, dtype='object')

    # Handle flat surfaces (exact 270.0 as flat)
    df.loc[aspect == 270.0, 'ORIENT'] = 'Flat'

    # Handle directional bins (excluding flat)
    bins = [0, 22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]
    labels = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']

    # Mask to apply binning only to non-flat areas
    directional_mask = (aspect != 270.0)
    df.loc[directional_mask, 'ORIENT'] = pd.cut(
        aspect[directional_mask],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
        ordered=False
    )

    return df

def map_wcm_classes(df, column='WCM'):
    """
    Map ESA WCM numeric land cover codes to descriptive class names.
    """
    wcm_class_map = {
        10.0: "Tree cover",
        20.0: "Shrubland",
        30.0: "Grassland",
        40.0: "Cropland",
        50.0: "Built-up",
        60.0: "Bare / sparse vegetation",
        70.0: "Snow and ice",
        80.0: "Permanent water bodies",
        90.0: "Herbaceous wetland",
        95.0: "Mangroves ",
        100.0: "Moss and lichen"
    }

    df['WCM_LABEL'] = df[column].map(wcm_class_map)
    return df


def map_wcm_classes_no_drop(df, column='WCM'):
    """
    Map ESA WCM numeric land cover codes to descriptive class names.
    """
    wcm_class_map = {
        0.0: "No Data",
        10.0: "Tree cover",
        20.0: "Shrubland",
        30.0: "Grassland",
        40.0: "Cropland",
        50.0: "Built-up",
        60.0: "Bare / sparse vegetation",
        70.0: "Snow and ice",
        80.0: "Permanent water bodies",
        90.0: "Herbaceous wetland",
        95.0: "Mangroves ",
        100.0: "Moss and lichen"
    }

    df['WCM_LABEL'] = df[column].map(wcm_class_map)
    return df

def map_wcm_classes_updated(df, column='WCM'):
    """
    Map ESA WCM numeric land cover codes to descriptive class names.
    """
    wcm_class_map = {
        10.0: "Tree cover",
        20.0: "Shrubland",
        30.0: "Grassland",
        40.0: "Cropland",
        50.0: "Built-up",
        60.0: "Bare / sparse vegetation",
        70.0: "Snow and ice",
        80.0: "Permanent water bodies",
        90.0: "Herbaceous wetland",
        95.0: "Mangroves ",
        100.0: "Moss and lichen"
    }

    df = df.copy()

    # Drop rows where WCM is 0.0 (No Data)
    df = df[df[column] != 0.0]

    df['WCM_LABEL'] = df[column].map(wcm_class_map)

    return df

def DEM_filled(df):
    df = df.copy()

    # Replace negative values in DEM_cop
    df['DEM_cop_filled'] = df['DEM_cop'].where(df['DEM_cop'] >= 0, df['DEM_mer'])
    df['DEM_cop_filled'] = df['DEM_cop_filled'].apply(lambda x: x if x >= 0 else 0)

    return df

def clean_numeric_features(df):
    df = df.copy()

    # Replace negative values in DEM_cop
    df['DEM_cop_filled'] = df['DEM_cop'].where(df['DEM_cop'] >= 0, df['DEM_mer'])
    df['DEM_cop_filled'] = df['DEM_cop_filled'].apply(lambda x: x if x >= 0 else 0)

    # Drop rows where WOP > 100
    df = df[df['WOP'] <= 100].reset_index(drop=True)

    return df.astype(np.float32) # Ensure all columns are float32

def clean_numeric_features_no_drop(df):
    df = df.copy()

    # Replace negative values in DEM_cop
    df['DEM_cop_filled'] = df['DEM_cop'].where(df['DEM_cop'] >= 0, df['DEM_mer'])
    df['DEM_cop_filled'] = df['DEM_cop_filled'].apply(lambda x: x if x >= 0 else 0)

    # Drop rows where WOP > 100
    # df = df[df['WOP'] <= 100].reset_index(drop=True)

    return df.astype(np.float32) # Ensure all columns are float32

def drop_bad_rows(df):
    df = df.copy()

    # Drop rows with 'NoData' in WCM
    df = df[df['WCM'] != 0.0]

    # Drop rows where WOP > 100
    df = df[df['WOP'] <= 100]

    # Optional: drop other NaNs
    # df = df.dropna()

    return df.reset_index(drop=True)

def drop_NoData_rows(X, y):
    X = X.copy()
    y = y.copy()

    # Boolean mask for rows to keep
    mask = (X['WCM'] != 0.0) & (X['WOP'] <= 100)

    # Apply mask to X and y
    X = X[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    return X, y

# Funcion que abre un archivos determinado y lo extrae en x e y
def process_img(n):
    import rasterio as rio
    import os

    # Paths
    image_path = f"C:/Users/maria/Documents/01_Teledeteccion/04_AEI/00_Practicas/Trabajo/Inputs/train/images/{n}.tif"
    label_path = f"C:/Users/maria/Documents/01_Teledeteccion/04_AEI/00_Practicas/Trabajo/Inputs/train/labels/{n}.png"

    # Load image and label
    img = rio.open(image_path)
    lbl = rio.open(label_path)

    row, col = img.height, img.width

    # Read bands
    b1 = img.read(1).astype('float32')  # VV
    b2 = img.read(2).astype('float32')  # VH
    b3 = img.read(3).astype('float32')  # DEM_mer
    b4 = img.read(4).astype('float32')  # DEM_cop
    b5 = img.read(5).astype('float32')  # WCM
    b6 = img.read(6).astype('float32')  # WOP

    # ---- Compute SAR ratios ----
    rat_vh_vv, rat_vv_vh, rat_norm = compute_sar_ratios(b1, b2)

    # ---- Compute terrain features ----
    slope = compute_slope_img(b4)
    aspect = compute_aspect_img(b4)

    # ---- Stack and flatten features ----
    feature_stack = {
        'VV': b1,
        'VH': b2,
        'VH_VV': rat_vh_vv,
        'VV_VH': rat_vv_vh,
        'NORM': rat_norm,
        'DEM_mer': b3,
        'DEM_cop': b4,
        'SLOPE': slope,
        'ASPECT': aspect,
        'WCM': b5,
        'WOP': b6
    }

    flat_feats = flatten_and_stack(feature_stack)
    df = pd.DataFrame(flat_feats).astype('float32')

    # ---- Load and flatten label ----
    label = lbl.read(1).astype('float32').flatten()

    return df, label, row, col
