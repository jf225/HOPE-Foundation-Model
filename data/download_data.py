import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


folder_path = Path("data") / "raw"


# The five Monash datasets KairosHope pretrains on and associated file number
monash_datasets = {
    "m4_hourly_dataset": 4656589,
    "tourism_monthly_dataset": 4656096,
    "electricity_hourly_dataset": 4656140,
    "traffic_hourly_dataset": 4656132,
    "weather_dataset": 4654822,
}


# The 45 UCR datasets in KairosHope testing
ucr_datasets = [
    "ArrowHead", "CBF", "ChlorineConcentration", "Coffee",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW",
    "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices",
    "FaceFour", "GunPoint", "ItalyPowerDemand", "Lightning7",
    "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW",
    "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2",
    "PhalangesOutlinesCorrect", "Plane",
    "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW",
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "Strawberry", "SyntheticControl",
    "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns",
    "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ", "Wafer", "Wine", "Worms", "WormsTwoClass", "Yoga",
]


def download_and_unzip(url, path):
    # Download and unzip file
    with urllib.request.urlopen(url) as response:
        zip_bytes = response.read()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zip_file:
        zip_file.extractall(path)


def download_monash():
    folder = folder_path / "monash"
    folder.mkdir(parents=True, exist_ok=True)

    for name, record in monash_datasets.items():
        print("Downloading:", name)
        url = f"https://zenodo.org/record/{record}/files/{name}.zip"
        download_and_unzip(url, folder)


def download_ucr():
    folder = folder_path / "ucr"
    folder.mkdir(parents=True, exist_ok=True)

    for name in ucr_datasets:
        print("Downloading:", name)
        url = f"https://timeseriesclassification.com/aeon-toolkit/{name}.zip"
        download_and_unzip(url, folder / name)


download_monash()
download_ucr()


window = 256
patch = 8
min_length = 64


def read_data_lines(path):
    lines = path.read_text(encoding="latin-1").splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower() == "@data":
            return [line for line in lines[i + 1:] if line.strip()]


def read_monash(path):
    all_series = []
    for line in read_data_lines(path):
        values = line.split(":")[-1].replace("?", "nan").split(",")
        all_series.append(np.array(values, dtype=np.float32))
    return all_series


def read_ucr(path):
    x, y = [], []
    for line in read_data_lines(path):
        values, label = line.rsplit(":", 1)
        x.append(values.split(","))
        y.append(label.strip())
    return np.array(x, dtype=np.float32), y


def cut_windows(series):
    windows = []

    # Take full 256 reading windows starting from most recent
    while len(series) >= window:
        windows.append(series[-window:])
        series = series[:-window]

    # Keep the leftover if it is long enough cut to patch size
    leftover = len(series) // patch * patch
    if leftover >= min_length:
        windows.append(series[-leftover:])

    return windows


def build_pretrain():
    x, valid_length = [], []

    for name in monash_datasets:
        for series in read_monash(folder_path / "monash" / f"{name}.tsf"):
            for w in cut_windows(series):
                # Skip windows with missing values or no change at all
                # I think with the kairoshope pretrain loss functions mean any flat values 
                if np.isnan(w).any() or w.max() == w.min():
                    continue

                # Pad short windows with zeros so array length matches up
                # Need to decide if padding is something we want to keep or just drop the window
                x.append(np.pad(w, (window - len(w), 0)))
                valid_length.append(len(w))

    np.savez("data/pretrain.npz", x=np.array(x), valid_length=np.array(valid_length))


def fix_length(x):
    length = x.shape[1]

    # Long series get shrunk down to 256 readings
    # Should probably test in the future if multiple windows of 256 is better or shrinking is better
    if length > window:
        old_positions = np.arange(length)
        new_positions = np.linspace(0, length - 1, window)
        return np.array([np.interp(new_positions, old_positions, row) for row in x], dtype=np.float32)

    # Dropping oldest data to fill patch length
    return x[:, length % patch:]


def build_ucr():
    folder = Path("data") / "ucr"
    folder.mkdir(exist_ok=True)

    for name in ucr_datasets:
        x_train, y_train = read_ucr(folder_path / "ucr" / name / f"{name}_TRAIN.ts")
        x_test, y_test = read_ucr(folder_path / "ucr" / name / f"{name}_TEST.ts")

        label_ids = {label: i for i, label in enumerate(sorted(set(y_train)))}
        y_train = np.array([label_ids[label] for label in y_train])
        y_test = np.array([label_ids[label] for label in y_test])

        np.savez(folder / f"{name}.npz", x_train=fix_length(x_train), y_train=y_train,
                 x_test=fix_length(x_test), y_test=y_test)


build_pretrain()
build_ucr()