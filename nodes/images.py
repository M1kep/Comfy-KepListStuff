import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Optional, Callable, TYPE_CHECKING

import torch
from PIL import ImageFont, ImageDraw, Image
import matplotlib.font_manager as fm
from torch import Tensor

from custom_nodes.Comfy_KepListStuff.utils import (
    zip_with_fill,
    tensor2pil,
    pil2tensor,
)
if TYPE_CHECKING:
    from mypy.typeshed.stdlib._typeshed import SupportsDunderGT, SupportsDunderLT


class ImageLabelOverlay:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "float_labels": ("FLOAT", {"forceInput": True}),
                "int_labels": ("INT", {"forceInput": True}),
                "str_labels": ("STRING", {"forceInput": True}),
            },
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Images",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "put_overlay"

    CATEGORY = "List Stuff"

    def put_overlay(
            self,
            images: List[Tensor],
            float_labels: Optional[List[float]] = None,
            int_labels: Optional[List[int]] = None,
            str_labels: Optional[List[str]] = None,
    ) -> Tuple[List[Tensor]]:
        batches = images

        labels_to_check: Dict[str, Union[List[float], List[int], List[str], None]] = {
            "float": float_labels if float_labels is not None else None,
            "int": int_labels if int_labels is not None else None,
            "str": str_labels if str_labels is not None else None
        }

        for l_type, labels in labels_to_check.items():
            if labels is None:
                continue
            if len(batches) != len(labels) and len(labels) != 1:
                raise Exception(
                    f"Non-matching input sizes got {len(batches)} Image Batches, {len(labels)} Labels for label type {l_type}"
                )

        image_h, _, _ = batches[0][0].size()

        font = ImageFont.truetype(fm.findfont(fm.FontProperties()), 60)

        ret_images: List[Tensor]= []
        loop_gen = zip_with_fill(batches, float_labels, int_labels, str_labels)
        for b_idx, (img_batch, float_lbl, int_lbl, str_lbl) in enumerate(loop_gen):
            batch: List[Tensor] = []
            for i_idx, img in enumerate(img_batch):
                pil_img = tensor2pil(img)
                # print(f"Batch: {b_idx} | img: {i_idx}")
                # print(img.size())
                draw = ImageDraw.Draw(pil_img)

                draw.text((0, image_h - 60), f"B: {b_idx} | I: {i_idx}", fill="red", font=font)

                y_offset = 0
                for _, lbl in zip(["float", "int", "str"], [float_lbl, int_lbl, str_lbl]):
                    if lbl is None:
                        continue
                    draw.rectangle((0, 0 + y_offset, 512, 60 + y_offset), fill="#ffff33")
                    draw.text((0, 0 + y_offset), str(lbl), fill="red", font=font)
                    y_offset += 60
                batch.append(pil2tensor(pil_img))

            ret_images.append(torch.cat(batch))

        return (ret_images,)


# Hack: string type that is always equal in not equal comparisons
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


# Our any instance wants to be a wildcard string
ANY = AnyType("*")

class XYImage:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "images": ("IMAGE",),
                "splits": ("INT", {"forceInput": True, "min": 1}),
                "flip_axis": (["False", "True"], {"default": "False"}),
                "batch_stack_mode": (["horizontal", "vertical"], {"default": "horizontal"}),
                "z_enabled": (["False", "True"], {"default": "False"}),
            },
            "optional": {
                "x_main_label": ("STRING", {}),
                "y_main_label": ("STRING", {}),
                "z_main_label": ("STRING", {}),
                "x_labels": (ANY,{}),
                "y_labels": (ANY,{}),
                "z_labels": (ANY,{}),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "xy_image"

    CATEGORY = "List Stuff"


    MAIN_LABEL_SIZE = 60
    LABEL_SIZE = 60
    Z_LABEL_SIZE = 60
    LABEL_COLOR = "#000"
    def xy_image(
            self,
            images: List[Tensor],
            splits: List[int],
            flip_axis: List[str],
            batch_stack_mode: List[str],
            z_enabled: List[str],
            x_main_label: Optional[List[str]] = None,
            y_main_label: Optional[List[str]] = None,
            z_main_label: Optional[List[str]] = None,
            x_labels: Optional[List[str]] = None,
            y_labels: Optional[List[str]] = None,
            z_labels: Optional[List[str]] = None,
    ) -> Tuple[List[Tensor]]:
        if len(flip_axis) != 1:
            raise Exception("Only single flip_axis value supported.")
        if len(batch_stack_mode) != 1:
            raise Exception("Only single batch stack mode supported.")
        if len(z_enabled) != 1:
            raise Exception("Only single z_enabled value supported.")
        if x_main_label is not None and len(x_main_label) != 1:
            raise Exception("Only single x_main_label value supported.")
        if y_main_label is not None and len(y_main_label) != 1:
            raise Exception("Only single y_main_label value supported.")
        if z_main_label is not None and len(z_main_label) != 1:
            raise Exception("Only single z_main_label value supported.")

        if x_main_label is not None and not isinstance(x_main_label[0], str):
            try:
                x_main_label[0] = str(x_main_label[0])
            except:
                raise Exception("x_main_label must be a string or convertible to a string.")
        if y_main_label is not None and not isinstance(y_main_label[0], str):
            try:
                y_main_label[0] = str(y_main_label[0])
            except:
                raise Exception("y_main_label must be a string or convertible to a string.")
        if z_main_label is not None and not isinstance(z_main_label[0], str):
            try:
                z_main_label[0] = str(z_main_label[0])
            except:
                raise Exception("z_main_label must be a string or convertible to a string.")

        if x_main_label is not None and x_main_label[0] == '':
            x_main_label = None
        if y_main_label is not None and y_main_label[0] == '':
            y_main_label = None
        if z_main_label is not None and z_main_label[0] == '':
            z_main_label = None

        stack_direction = "horizontal"
        if flip_axis[0] == "True":
            stack_direction = "vertical"
            x_labels, y_labels = y_labels, x_labels
            x_main_label, y_main_label = y_main_label, x_main_label

        batch_stack_direction = batch_stack_mode[0]

        if len(splits) == 1:
            splits = splits * (int(len(images) / splits[0]))
            if sum(splits) != len(images):
                splits.append(len(images) - sum(splits))
        else:
            if sum(splits) != len(images):
                raise Exception("Sum of splits must equal number of images.")

        batches = images
        batch_size = len(batches[0])

        # TODO: Some better way...
        # Currently chops splits to match x_labels/y_labels and then loops over the split set over and over
        num_z = 1
        splits_per_z = len(splits)
        images_per_z = len(images)
        if z_enabled[0] == "True":
            if y_labels is None or x_labels is None:
                raise Exception("Must provide x_labels and y_labels when z_enabled is True.")

            if stack_direction == "horizontal":
                splits_per_z = len(x_labels)
            else:
                splits_per_z = len(y_labels)

            num_z = int(len(splits) / splits_per_z)
            splits = splits[:splits_per_z]
            images_per_z = sum(splits)

        image_h, image_w, _ = batches[0][0].size()
        if batch_stack_direction == "horizontal":
            batch_h = image_h
            # stack horizontally
            batch_w = image_w * batch_size
        else:
            # stack vertically
            batch_h = image_h * batch_size
            batch_w = image_w

        if stack_direction == "horizontal":
            full_w = batch_w * len(splits)
            full_h = batch_h * max(splits)
        else:
            full_w = batch_w * max(splits)
            full_h = batch_h * len(splits)
        grid_w = full_w
        _ = full_h

        y_label_offset = 0
        has_horizontal_labels = False
        if x_labels is not None:
            x_labels = [str(lbl) for lbl in x_labels]
            if stack_direction == "horizontal":
                if len(x_labels) != len(splits):
                    raise Exception("Number of horizontal labels must match number of splits.")
            else:
                if len(x_labels) != max(splits):
                    raise Exception("Number of horizontal labels must match maximum split size.")
            full_h += self.LABEL_SIZE
            y_label_offset = self.LABEL_SIZE
            has_horizontal_labels = True

        x_label_offset = 0
        has_vertical_labels = False
        if y_labels is not None:
            y_labels = [str(lbl) for lbl in y_labels]
            if stack_direction == "horizontal":
                if len(y_labels) != max(splits):
                    raise Exception(f"Number of vertical labels must match maximum split size. Got {len(y_labels)} labels for {max(splits)} splits.")
            else:
                if len(y_labels) != len(splits):
                    raise Exception(f"Number of vertical labels must match number of splits. Got {len(y_labels)} labels for {len(splits)} splits.")
            full_w += self.LABEL_SIZE
            x_label_offset = self.LABEL_SIZE
            has_vertical_labels = True

        has_z_labels = False
        if z_labels is not None:
            has_z_labels = True
            z_labels = [str(lbl) for lbl in z_labels]
            if z_main_label is not None:
                z_labels = [f"{z_main_label[0]}: {lbl}" for lbl in z_labels]
            full_h += self.Z_LABEL_SIZE
            y_label_offset += self.Z_LABEL_SIZE
            if len(z_labels) != num_z:
                raise Exception(f"Number of z_labels must match number of z splits. Got {len(z_labels)} labels for {num_z} splits.")

        has_main_x_label = False
        if x_main_label is not None:
            full_h += self.MAIN_LABEL_SIZE
            y_label_offset += self.MAIN_LABEL_SIZE
            has_main_x_label = True

        has_main_y_label = False
        if y_main_label is not None:
            full_w += self.MAIN_LABEL_SIZE
            x_label_offset += self.MAIN_LABEL_SIZE
            has_main_y_label = True

        images = []
        for z_idx in range(num_z):
            full_image = Image.new("RGB", (full_w, full_h))
            full_draw = ImageDraw.Draw(full_image)

            full_draw.rectangle((0, 0, full_w, full_h), fill="#ffffff")

            batch_idx = 0
            active_y_offset = 0
            active_x_offset = 0
            if has_z_labels:
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), self.Z_LABEL_SIZE)
                full_draw.rectangle((0, 0, full_w, self.Z_LABEL_SIZE), fill="#ffffff")
                full_draw.text((grid_w//2 + x_label_offset, 0),  z_labels[z_idx], anchor='ma', fill=self.LABEL_COLOR, font=font)
                active_y_offset += self.Z_LABEL_SIZE

            if has_main_x_label:
                assert x_main_label is not None
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), self.MAIN_LABEL_SIZE)
                full_draw.rectangle((0, active_y_offset, full_w, self.MAIN_LABEL_SIZE + active_y_offset), fill="#ffffff")
                full_draw.text((grid_w//2 + x_label_offset, 0 + active_y_offset), x_main_label[0], anchor='ma', fill=self.LABEL_COLOR, font=font)
                active_y_offset += self.MAIN_LABEL_SIZE

            if has_horizontal_labels:
                assert x_labels is not None
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), self.LABEL_SIZE)
                for label_idx, label in enumerate(x_labels):
                    x_offset = (batch_w * label_idx) + x_label_offset
                    full_draw.rectangle((x_offset, 0 + active_y_offset, x_offset + batch_w, self.LABEL_SIZE + active_y_offset), fill="#ffffff")
                    full_draw.text((x_offset + (batch_w / 2), 0 + active_y_offset), label, anchor='ma', fill=self.LABEL_COLOR, font=font)

            if has_main_y_label:
                assert y_main_label is not None
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), self.MAIN_LABEL_SIZE)

                img_txt = Image.new('RGB', (full_h - active_y_offset, self.MAIN_LABEL_SIZE))
                draw_txt = ImageDraw.Draw(img_txt)
                draw_txt.rectangle((0, 0, full_h - active_y_offset, self.MAIN_LABEL_SIZE), fill="#ffffff")
                draw_txt.text(((full_h - active_y_offset)//2, 0),  y_main_label[0], anchor='ma', fill=self.LABEL_COLOR, font=font)
                img_txt = img_txt.rotate(90, expand=True)
                full_image.paste(img_txt, (active_x_offset, active_y_offset))
                active_x_offset += self.MAIN_LABEL_SIZE

            if has_vertical_labels:
                assert y_labels is not None
                font = ImageFont.truetype(fm.findfont(fm.FontProperties()), self.LABEL_SIZE)
                for label_idx, label in enumerate(y_labels):
                    y_offset = (batch_h * label_idx) + y_label_offset

                    img_txt = Image.new('RGB', (batch_h, self.LABEL_SIZE))
                    draw_txt = ImageDraw.Draw(img_txt)
                    draw_txt.rectangle((0, 0, batch_h, self.LABEL_SIZE), fill="#ffffff")
                    draw_txt.text((batch_h//2, 0),  label, anchor='ma', fill=self.LABEL_COLOR, font=font)
                    img_txt = img_txt.rotate(90, expand=True)
                    full_image.paste(img_txt, (active_x_offset, y_offset))

            for split_idx, split in enumerate(splits):
                for idx_in_split in range(split):
                    batch_img = Image.new("RGB", (batch_w, batch_h))
                    batch = batches[batch_idx + idx_in_split + images_per_z * z_idx]
                    if batch_stack_direction == "horizontal":
                        for img_idx, img in enumerate(batch):
                            x_offset = image_w * img_idx
                            batch_img.paste(tensor2pil(img), (x_offset, 0))
                    else:
                        for img_idx, img in enumerate(batch):
                            y_offset = image_h * img_idx
                            batch_img.paste(tensor2pil(img), (0, y_offset))

                    if stack_direction == "horizontal":
                        x_offset = batch_w * split_idx + x_label_offset
                        y_offset = batch_h * idx_in_split + y_label_offset
                    else:
                        x_offset = batch_w * idx_in_split + x_label_offset
                        y_offset = batch_h * split_idx + y_label_offset
                    full_image.paste(batch_img, (x_offset, y_offset))

                batch_idx += split
            images.append(pil2tensor(full_image))
        return (images,)

class VariableImageBuilder:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "r": ("INT", {"defaultInput": True, "min": 0, "max": 255}),
                "g": ("INT", {"defaultInput": True, "min": 0, "max": 255}),
                "b": ("INT", {"defaultInput": True, "min": 0, "max": 255}),
                "a": ("INT", {"defaultInput": True, "min": 0, "max": 255}),
                "width": ("INT", {"defaultInput": False, "default": 512}),
                "height": ("INT", {"defaultInput": False, "default": 512}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
            },
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "generate_images"

    CATEGORY = "List Stuff"

    def generate_images(
            self,
            r: int,
            g: int,
            b: int,
            a: int,
            width: int,
            height: int,
            batch_size: int,
    ) -> Tuple[Tensor]:
        batch_tensors: List[Tensor] = []
        for _ in range(batch_size):
            image = Image.new("RGB", (width, height), color=(r, g, b, a))
            batch_tensors.append(pil2tensor(image))
        return (torch.cat(batch_tensors),)


class EmptyImages:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {},
            "optional": {
                "num_images": ("INT", {"forceInput": True, "min": 1}),
                "splits": ("INT", {"forceInput": True, "min": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1}),
            }
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Image",)
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "generate_empty_images"

    CATEGORY = "List Stuff"

    def generate_empty_images(
            self,
            num_images: Optional[List[int]] = None,
            splits: Optional[List[int]] = None,
            batch_size: Optional[List[int]] = None,
    ) -> Tuple[List[Tensor]]:
        if batch_size is None:
            batch_size = [1]
        else:
            if len(batch_size) != 1:
                raise Exception("Only single batch size supported.")

        if num_images is None and splits is None:
            raise Exception("Must provide either num_images or splits.")

        if num_images is not None and len(num_images) != 1:
            raise Exception("Only single num_images supported.")

        if num_images is not None and splits is None:
            # If splits is None, then all images are in one split
            splits = [num_images[0]]

        if num_images is None and splits is not None:
            # If num_images is None, then it should be the sum of all splits
            num_images = [sum(splits)]

        if num_images is not None and splits is not None:
            if len(splits) == 1:
                # Fill splits with same value enough times to sum to num_images
                fills = int(num_images[0] / splits[0])
                splits = [splits[0]] * fills
                if sum(splits) != num_images[0]:
                    splits.append(num_images[0] - sum(splits))
            else:
                if sum(splits) != num_images[0]:
                    raise Exception("Sum of splits must match number of images.")

        if splits is None:
            raise ValueError("Unexpected error: Splits is None")

        ret_images: List[Tensor] = []
        for split_idx, split in enumerate(splits):
            # Rotate between fully dynamic range of colors
            base_color = (
                50 + (split_idx * 45) % 200,  # Cycle between 50 and 250
                30 + (split_idx * 75) % 200,
                10 + (split_idx * 105) % 200,
            )
            print(f"Splits: {split} | Base Color: {base_color}")

            for _ in range(split):
                batch_tensor = torch.zeros(batch_size[0], 512, 512, 3)
                for batch_idx in range(batch_size[0]):
                    batch_color = (
                        (base_color[0] + int(((255 - base_color[0]) / batch_size[0]) * batch_idx)),
                        (base_color[1] + int(((255 - base_color[1]) / batch_size[0]) * batch_idx)),
                        (base_color[2] + int(((255 - base_color[2]) / batch_size[0]) * batch_idx)),
                    )
                    image = Image.new("RGB", (512, 512), color=batch_color)
                    batch_tensor[batch_idx] = pil2tensor(image)
                ret_images.append(batch_tensor)
        return (ret_images,)

class ImageListLoader:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "folder_path": ("STRING", {}),
                "file_filter": ("STRING", {"default": "*.png"}),
                "sort_method": (["numerical", "alphabetical"], {"default": "numerical"}),
            },
        }

    RELOAD_INST = True
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("Images",)
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_images"

    CATEGORY = "List Stuff"

    @staticmethod
    def numerical_sort(file_name: Path) -> int:
        subbed = re.sub("\D", "", str(file_name))
        if subbed == "":
            return 0
        return int(subbed)
    
    
    @staticmethod
    def alphabetical_sort(file_name: Path) -> str:
        return str(file_name)

    def load_images(
        self, folder_path: str, file_filter: str, sort_method: str
    ) -> Tuple[List[Tensor]]:
        folder = Path(folder_path)
    
        if not folder.is_dir():
            raise Exception(f"Folder path {folder_path} does not exist.")

        sort_method_impl: Callable[[str], Union[SupportsDunderGT, SupportsDunderLT]]
        if sort_method == "numerical":
            sort_method_impl = self.numerical_sort
        elif sort_method == "alphabetical":
            sort_method_impl = self.alphabetical_sort
        else:
            raise ValueError(f"Unknown sort method {sort_method}")

        files = sorted(folder.glob(file_filter), key=sort_method_impl)
        images = [pil2tensor(Image.open(file)) for file in files]
    
        return (images,)
