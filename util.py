import monai
import numpy as np
from typing import Mapping, Hashable, List

# Custom transform to convert a pair of segmentation mask arrays to a single one-hot encoded array
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

