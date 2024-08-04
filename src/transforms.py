import torch


class MinMaxNormalizer:
    def __call__(self, tensor):
        # Check if the tensor has any elements to avoid division by zero
        if tensor.numel() == 0:
            raise ValueError("Input tensor has no elements")

        min_val = torch.min(tensor)
        max_val = torch.max(tensor)

        if min_val == max_val:
            raise ValueError("Input tensor has constant value, cannot normalize")

        # Normalize the tensor to [0, 1]
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor
