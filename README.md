# DALI GPU video dataloader working examples

# Requirements

nvidia-dali >= 0.18  
pytorch >= 0.4.0

# Usage

```bash
$ python single_gpu_uid_label_loader.py
Total number of samples: 65
input shape: torch.Size([2, 3, 16, 224, 224])
video uids:
tensor([[31969],
        [31961]], device='cuda:0', dtype=torch.int32)
labels:
tensor([[1],
        [0]], device='cuda:0', dtype=torch.int32)
frame nums:
tensor([[32],
        [48]], device='cuda:0', dtype=torch.int32)
x crop pos:
tensor([[0.9071],
        [0.6472]])
y crop pos:
tensor([[0.5866],
        [0.6747]])
is flipped:
tensor([[0],
        [1]], dtype=torch.int32)
```
