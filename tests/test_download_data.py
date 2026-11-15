import io
import zipfile

import numpy as np
import pytest

from data import download_data as data

# --------------------------------------------------
# read_data_lines
# --------------------------------------------------


def test_read_data_lines(tmp_path):
    """Only lines after @data should be returned."""
    file = tmp_path / "sample.ts"
    file.write_text(
        "@problemName Example\n"
        "@timestamps false\n"
        "@data\n"
        "1,2,3:classA\n"
        "4,5,6:classB\n",
        encoding="latin-1",
    )

    lines = data.read_data_lines(file)

    assert lines == ["1,2,3:classA", "4,5,6:classB"]


def test_read_data_lines_ignores_blank_lines(tmp_path):
    file = tmp_path / "sample.ts"
    file.write_text("@data\n" "\n" "1,2,3:classA\n" "  \n" "4,5,6:classB\n")

    lines = data.read_data_lines(file)

    assert lines == ["1,2,3:classA", "4,5,6:classB"]


def test_read_data_lines_without_data_tag(tmp_path):
    """A file without @data should not return data rows."""
    file = tmp_path / "sample.ts"
    file.write_text("@problemName Example\n@classLabel true\n")

    assert data.read_data_lines(file) is None


# --------------------------------------------------
# read_monash
# --------------------------------------------------


def test_read_monash(tmp_path):
    """Monash rows should be converted into float arrays."""
    file = tmp_path / "sample.tsf"
    file.write_text(
        "@attribute series_name string\n"
        "@attribute start_timestamp date\n"
        "@frequency hourly\n"
        "@data\n"
        "series1:2020-01-01 00-00-00:1.0,2.0,3.0\n"
        "series2:2020-01-02 00-00-00:4.0,5.0,6.0\n"
    )

    result = data.read_monash(file)

    assert len(result) == 2
    np.testing.assert_array_equal(
        result[0], np.array([1.0, 2.0, 3.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        result[1], np.array([4.0, 5.0, 6.0], dtype=np.float32)
    )


def test_read_monash_missing_values(tmp_path):
    """Missing values represented by ? should become NaN."""
    file = tmp_path / "sample.tsf"
    file.write_text("@data\n" "series1:2020-01-01 00-00-00:1.0,?,3.0\n")

    result = data.read_monash(file)

    assert np.isnan(result[0][1])


# --------------------------------------------------
# read_ucr
# --------------------------------------------------


def test_read_ucr(tmp_path):
    """UCR rows should be separated into features and labels."""
    file = tmp_path / "sample.ts"
    file.write_text(
        "@problemName Example\n" "@data\n" "1.0,2.0,3.0:classA\n" "4.0,5.0,6.0:classB\n"
    )

    x, y = data.read_ucr(file)

    assert x.shape == (2, 3)
    np.testing.assert_array_equal(x, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    assert y == ["classA", "classB"]


def test_read_ucr_preserves_labels_with_colons(tmp_path):
    """The final colon should separate the class label from readings."""
    file = tmp_path / "sample.ts"
    file.write_text("@data\n" "1.0,2.0:class:A\n")

    x, y = data.read_ucr(file)

    np.testing.assert_array_equal(x, [[1.0, 2.0]])
    assert y == ["class:A"]


# --------------------------------------------------
# cut_windows
# --------------------------------------------------


def test_cut_windows_exact_window():
    """A series of exactly 256 readings should produce one window."""
    series = np.arange(256, dtype=np.float32)

    windows = data.cut_windows(series)

    assert len(windows) == 1
    assert len(windows[0]) == 256
    np.testing.assert_array_equal(windows[0], series)


def test_cut_windows_multiple_windows():
    """Long series should be split into 256-reading windows."""
    series = np.arange(512, dtype=np.float32)

    windows = data.cut_windows(series)

    assert len(windows) == 2
    assert len(windows[0]) == 256
    assert len(windows[1]) == 256

    # Windows should be taken from newest to oldest.
    np.testing.assert_array_equal(windows[0], series[256:])
    np.testing.assert_array_equal(windows[1], series[:256])


def test_cut_windows_keeps_valid_leftover():
    """A leftover of at least 64 readings should be retained."""
    series = np.arange(320, dtype=np.float32)

    windows = data.cut_windows(series)

    assert len(windows) == 2
    assert len(windows[0]) == 256
    assert len(windows[1]) == 64


def test_cut_windows_drops_short_leftover():
    """A leftover shorter than 64 readings should be dropped."""
    series = np.arange(300, dtype=np.float32)

    windows = data.cut_windows(series)

    assert len(windows) == 1
    assert len(windows[0]) == 256


def test_cut_windows_truncates_leftover_to_patch_size():
    """Leftover readings should be truncated to a multiple of 8."""
    series = np.arange(326, dtype=np.float32)

    windows = data.cut_windows(series)

    assert len(windows) == 2
    assert len(windows[1]) == 64


# --------------------------------------------------
# fix_length
# --------------------------------------------------


def test_fix_length_long_series():
    """Long series should be interpolated to 256 readings."""
    x = np.arange(512, dtype=np.float32).reshape(1, 512)

    result = data.fix_length(x)

    assert result.shape == (1, 256)
    assert result.dtype == np.float32
    assert result[0, 0] == pytest.approx(0.0)
    assert result[0, -1] == pytest.approx(511.0)


def test_fix_length_short_series():
    """Short series should have incomplete readings removed."""
    x = np.arange(70, dtype=np.float32).reshape(1, 70)

    result = data.fix_length(x)

    assert result.shape == (1, 64)
    np.testing.assert_array_equal(result[0], np.arange(6, 70))


def test_fix_length_already_patch_aligned():
    """A patch-aligned short series should remain unchanged."""
    x = np.arange(64, dtype=np.float32).reshape(1, 64)

    result = data.fix_length(x)

    assert result.shape == (1, 64)
    np.testing.assert_array_equal(result, x)


# --------------------------------------------------
# download_and_unzip
# --------------------------------------------------


def test_download_and_unzip(tmp_path, monkeypatch):
    """A downloaded ZIP should be extracted into the target folder."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        zip_file.writestr("sample.txt", "test contents")

    zip_bytes = zip_buffer.getvalue()

    class MockResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return zip_bytes

    monkeypatch.setattr(data.urllib.request, "urlopen", lambda url: MockResponse())

    data.download_and_unzip("https://example.com/sample.zip", tmp_path)

    extracted = tmp_path / "sample.txt"

    assert extracted.exists()
    assert extracted.read_text() == "test contents"


# --------------------------------------------------
# download URL construction
# --------------------------------------------------


def test_download_monash_urls(tmp_path, monkeypatch):
    """Each Monash dataset should use its expected download URL."""
    monkeypatch.setattr(data, "folder_path", tmp_path / "raw")

    calls = []

    def mock_download(url, path):
        calls.append((url, path))

    monkeypatch.setattr(data, "download_and_unzip", mock_download)

    data.download_monash()

    assert len(calls) == len(data.monash_datasets)

    for name, record in data.monash_datasets.items():
        expected_url = f"https://zenodo.org/record/{record}/files/{name}.zip"

        assert any(url == expected_url for url, _ in calls)


def test_download_ucr_urls(tmp_path, monkeypatch):
    """Each UCR dataset should use its expected download URL."""
    monkeypatch.setattr(data, "folder_path", tmp_path / "raw")

    calls = []

    def mock_download(url, path):
        calls.append((url, path))

    monkeypatch.setattr(data, "download_and_unzip", mock_download)

    data.download_ucr()

    assert len(calls) == len(data.ucr_datasets)

    for name in data.ucr_datasets:
        expected_url = (
            f"https://timeseriesclassification.com/" f"aeon-toolkit/{name}.zip"
        )

        assert any(url == expected_url for url, _ in calls)


# --------------------------------------------------
# build_pretrain
# --------------------------------------------------


def test_build_pretrain(tmp_path, monkeypatch):
    """Pretraining data should be saved with padding and valid lengths."""
    raw_folder = tmp_path / "raw"
    monash_folder = raw_folder / "monash"
    monash_folder.mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "folder_path", raw_folder)
    monkeypatch.setattr(data, "monash_datasets", {"sample_dataset": 123})

    series = np.arange(64, dtype=np.float32) + 1

    monkeypatch.setattr(data, "read_monash", lambda path: [series])

    data.build_pretrain()

    saved = np.load(tmp_path / "data" / "pretrain.npz")

    assert saved["x"].shape == (1, 256)
    assert saved["valid_length"].tolist() == [64]

    # Padding should be at the beginning.
    np.testing.assert_array_equal(saved["x"][0, :192], np.zeros(192))
    np.testing.assert_array_equal(saved["x"][0, 192:], series)


def test_build_pretrain_skips_invalid_series(tmp_path, monkeypatch):
    """Flat or NaN-containing windows should be skipped."""
    raw_folder = tmp_path / "raw"
    (raw_folder / "monash").mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "folder_path", raw_folder)
    monkeypatch.setattr(data, "monash_datasets", {"sample_dataset": 123})

    flat = np.ones(256, dtype=np.float32)
    with_nan = np.arange(256, dtype=np.float32)
    with_nan[10] = np.nan

    monkeypatch.setattr(data, "read_monash", lambda path: [flat, with_nan])

    data.build_pretrain()

    saved = np.load(tmp_path / "data" / "pretrain.npz")

    assert saved["x"].shape == (0, 256)
    assert saved["valid_length"].shape == (0,)


# --------------------------------------------------
# build_ucr
# --------------------------------------------------


def test_build_ucr(tmp_path, monkeypatch):
    """UCR data should be saved with contiguous training labels."""
    raw_folder = tmp_path / "raw"
    (raw_folder / "ucr" / "Sample").mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "folder_path", raw_folder)
    monkeypatch.setattr(data, "ucr_datasets", ["Sample"])

    x_train = np.arange(128, dtype=np.float32).reshape(2, 64)
    y_train = ["classB", "classA"]

    x_test = np.arange(64, dtype=np.float32).reshape(1, 64)
    y_test = ["classA"]

    def mock_read_ucr(path):
        if "TRAIN" in str(path):
            return x_train, y_train
        return x_test, y_test

    monkeypatch.setattr(data, "read_ucr", mock_read_ucr)

    data.build_ucr()

    saved = np.load(tmp_path / "data" / "ucr" / "Sample.npz")

    assert saved["x_train"].shape == (2, 64)
    assert saved["x_test"].shape == (1, 64)

    # Labels should be mapped using sorted training classes.
    assert saved["y_train"].tolist() == [1, 0]
    assert saved["y_test"].tolist() == [0]


def test_build_ucr_unknown_test_label(tmp_path, monkeypatch):
    """Test labels absent from training should be handled explicitly."""
    raw_folder = tmp_path / "raw"
    (raw_folder / "ucr" / "Sample").mkdir(parents=True)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "folder_path", raw_folder)
    monkeypatch.setattr(data, "ucr_datasets", ["Sample"])

    def mock_read_ucr(path):
        if "TRAIN" in str(path):
            return np.ones((1, 64)), ["classA"]
        return np.ones((1, 64)), ["classB"]

    monkeypatch.setattr(data, "read_ucr", mock_read_ucr)

    with pytest.raises(KeyError):
        data.build_ucr()
