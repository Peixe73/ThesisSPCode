from .csv_dataset import CSVDataset
from typing import Optional, Callable, TYPE_CHECKING, Literal
from core.datasets.csv_dataset import CSVDataset


import torch
import torchvision
import torchvision.transforms.v2 as transforms
from PIL import Image

from torchvision.transforms.v2 import Transform
from pathlib import Path
if TYPE_CHECKING:
    import pandas as pd

_IDENTITY_TRANSFORM : Transform = transforms.Lambda(lambda x : x)

class CSVImageDatasetGTSRB(CSVDataset):
    def __init__(
        self,
        csv_path: str | Path,
        images_path: str | Path,
        image_columns: list[str | tuple[str, Callable[[str], str]]],
        target: str | list[str],
        features: Optional[list[str]] = None,
        dtype: Optional[torch.dtype] = torch.float32,
        global_transform: Optional[Transform] = None,
        column_transforms: Optional[dict[str, Transform]] = None,
        shuffle: bool = True,
        random_state=None,
        splits: float | tuple[float, float] = (0.7, 0.15),
        filter: Optional[Callable[['pd.Series'], bool]] = None,

        # GTSRB-specific
        roi_columns: tuple[str, str, str, str] = ("Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"),
        image_size: tuple[int, int] = (128, 128),
    ):
        super().__init__(
            csv_path,
            target,
            features,
            shuffle=shuffle,
            random_state=random_state,
            splits=splits,
            filter=filter
        )

        self.images_path = Path(images_path)
        self.dtype = dtype
        self.global_transform: Transform = global_transform or _IDENTITY_TRANSFORM
        self.column_transforms = column_transforms or {}
        self.skip_image_loading: bool | Literal['get_path'] = False

        self.roi_columns = roi_columns
        self.image_size = image_size

        for column in image_columns:
            if isinstance(column, tuple):
                col_name, path_getter = column
            else:
                col_name = column
                path_getter = lambda x: x

            def image_getter(x, col_name=col_name):
                path: Path = self.images_path.joinpath(path_getter(x))

                # --- Skip logic ---
                if self.skip_image_loading:
                    if self.skip_image_loading == 'get_path':
                        return path.relative_to(self.images_path)
                    else:
                        return torch.full((1,), float('nan'))

                # --- Load image ---
                try:
                    image: torch.Tensor = torchvision.io.decode_image(path)
                except RuntimeError:
                    img = Image.open(path).convert("RGB")
                    image = transforms.functional.to_image(img)

                # --- Convert dtype ---
                if self.dtype is not None:
                    image = transforms.functional.to_dtype(
                        image, dtype=self.dtype, scale=True
                    )

                # --- Get ROI from CSV ---
                try:
                    row = self.data[self.data[col_name] == x].iloc[0]
                    x1 = int(row[self.roi_columns[0]])
                    y1 = int(row[self.roi_columns[1]])
                    x2 = int(row[self.roi_columns[2]])
                    y2 = int(row[self.roi_columns[3]])

                    # Crop to ROI
                    image = transforms.functional.crop(
                        image,
                        top=y1,
                        left=x1,
                        height=(y2 - y1),
                        width=(x2 - x1),
                    )
                except Exception:
                    # fallback: no crop
                    pass

                # --- Make square (center pad) ---
                _, h, w = image.shape
                size = max(h, w)

                pad_h = size - h
                pad_w = size - w

                padding = (
                    pad_w // 2,
                    pad_h // 2,
                    pad_w - pad_w // 2,
                    pad_h - pad_h // 2,
                )

                image = transforms.functional.pad(image, padding, fill=0)

                # --- Resize ---
                image = transforms.functional.resize(image, self.image_size)

                # --- Apply existing transforms ---
                image = self.global_transform(image)

                column_transform = self.column_transforms.get(
                    col_name, _IDENTITY_TRANSFORM
                )
                image = column_transform(image)

                return image

            self.add_column_preprocessor(
                col_name, image_getter, is_scalar=False
            )