import nibabel as nib
from pathlib import Path
import numpy as np


class NiiGZData:
    def __init__(self, file_path: str|Path):
        self.file_path = file_path
        self.nii = nib.load(file_path)

    @property
    def fdata(self) -> np.ndarray:
        return self.nii.get_fdata()
    
    def split(self, direction: int|None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]|np.ndarray:
        if direction is None:
            # Sagittal, Coronal, Axial
            return self.split(direction=0), self.split(direction=1), self.split(direction=2)
        return np.split(self.fdata, self.fdata.shape[direction], axis=direction)

    def select(self, slice_idx: int, direction: int = -1) -> np.ndarray:
        if direction < 0:
            direction += self.fdata.ndim
        if direction == 0:
            return self.fdata[slice_idx, :, :]
        elif direction == 1:
            return self.fdata[:, slice_idx, :]
        elif direction == 2:
            return self.fdata[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid direction: {direction}")

    def shape(self) -> tuple[int, int, int]:
        return self.fdata.shape
