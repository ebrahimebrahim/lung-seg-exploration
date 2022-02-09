import monai
import numpy as np
from typing import Mapping, Hashable, List, Sequence
from torch.utils.data._utils.collate import default_collate

# This module contains all the extra components needed for data loading and segmentation model training,
# some of which need to exist in any environment where the saved model is to be deployed.


# Custom transform to convert a collection of segmentation mask arrays to a single one-hot encoded array
class MasksToOneHotD(monai.transforms.MapTransform):
    def __init__(self, keys: monai.config.KeysCollection,
                 keyList: List[Hashable], newKeyName: str) -> None:
        super().__init__(keys)
        self.keyList = keyList
        self.newKeyName = newKeyName
        assert(len(keyList)>0)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:

        # (if this were to be contributed, I'd have to pay attention to whether keys are in data)
        # (also I'd want to raise more clear exceptions than these asserts)

        assert(all(key in self.keys for key in self.keyList))
        assert(all(key in data.keys() for key in self.keyList))
        assert(self.newKeyName not in data.keys())

        background_mask = (sum(data[key] for key in self.keyList)==0).astype('int8')

        # Assumes these were numpy arrays.
        # If they were torch tensors we'd have to do "torch.stack" and use argument "dim" instead of "axis"
        data[self.newKeyName] = np.stack(
            [background_mask] + [data[key] for key in self.keyList],
            axis=0
        )

        return data


# Custom transform to convert a collection of segmentation mask arrays to a single mask, one-hot encoded (e.g. union left and right lung into one thing)
# Always allows missing keys, just being a no-op if any keys are missing
class UnionMasksD(monai.transforms.MapTransform):
    def __init__(self, keys: monai.config.KeysCollection,
                 keyList: List[Hashable], newKeyName: str) -> None:
        super().__init__(keys)
        self.keyList = keyList
        self.newKeyName = newKeyName
        assert(len(keyList)>0)

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Mapping[Hashable, np.ndarray]:

        if not all(key in data.keys() for key in self.keyList):
            return data

        assert(all(key in self.keys for key in self.keyList))
        assert(self.newKeyName not in data.keys())

        sum_of_masks = sum(data[key] for key in self.keyList)
        background_mask = (sum_of_masks==0).astype('int8')
        foreground_mask = (sum_of_masks!=0).astype('int8')

        # Assumes these were numpy arrays.
        # If they were torch tensors we'd have to do "torch.stack" and use argument "dim" instead of "axis"
        data[self.newKeyName] = np.stack(
            [background_mask, foreground_mask],
            axis=0
        )

        return data

def rgb_to_grayscale(x):
    """Given a numpy array with shape (H,W,C), return "grayscale" one of shape (H,W);
    behaves as no-op if given array with shape already being (H,W)"""
    if len(x.shape)==2: return x
    elif len(x.shape)==3: return x.mean(axis=2)
    else: raise Exception("rgb_to_grayscale: unexpected number of axes in array")


# A custom version of list_data_collate copied straight out of MONAI codebase and slightly modified.
def list_data_collate_no_meta(batch: Sequence):
    """
    Enhancement for PyTorch DataLoader default collate.
    If dataset already returns a list of batch data that generated in transforms, need to merge all data to 1 list.
    Then it's same as the default collate behavior.

    Note:
        Need to use this collate if apply some transforms that can generate batch data.

    """
    elem = batch[0]
    data = [i for k in batch for i in k] if isinstance(elem, list) else batch
    key = None
    try:
        elem = batch[0]
        if isinstance(elem, Mapping):
            ret = {}
            for k in elem:
                key = k

                # Ebrahim's modification: skip any meta_dicts. Collating them is glitchy in my use-case.
                if "_meta_dict" in key: continue

                ret[k] = default_collate([d[k] for d in data])
            return ret
        return default_collate(data)
    except RuntimeError as re:
        re_str = str(re)
        if "equal size" in re_str:
            if key is not None:
                re_str += f"\nCollate error on the key '{key}' of dictionary data."
            re_str += (
                "\n\nMONAI hint: if your transforms intentionally create images of different shapes, creating your "
                + "`DataLoader` with `collate_fn=pad_list_data_collate` might solve this problem (check its "
                + "documentation)."
            )
        raise RuntimeError(re_str) from re
    # except TypeError as re:
    #     re_str = str(re)
    #     if "numpy" in re_str and "Tensor" in re_str:
    #         if key is not None:
    #             re_str += f"\nCollate error on the key '{key}' of dictionary data."
    #         re_str += (
    #             "\n\nMONAI hint: if your transforms intentionally create mixtures of torch Tensor and numpy ndarray, "
    #             + "creating your `DataLoader` with `collate_fn=pad_list_data_collate` might solve this problem "
    #             + "(check its documentation)."
    #         )
    #     raise TypeError(re_str) from re
