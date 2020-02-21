from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin import pytorch
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import torch
import numpy as np

class VideoReaderPipeline(Pipeline):
    def __init__(self, batch_size, sequence_length, num_threads, device_id, file_list, crop_size):
        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.reader = ops.VideoReader(device="gpu", file_list=file_list, sequence_length=sequence_length, normalized=True,
                                     random_shuffle=True, image_type=types.RGB, dtype=types.FLOAT, initial_fill=16, enable_frame_num=True, stride=1, step=-1)
        self.uniform = ops.Uniform(range=(0.0, 1.0))
        self.coin = ops.CoinFlip(probability=0.5)
#        self.crop = ops.Crop(device="gpu", crop=crop_size, output_dtype=types.FLOAT)
#        self.transpose = ops.Transpose(device="gpu", perm=[3, 0, 1, 2])
        self.cropmirrornorm = ops.CropMirrorNormalize(device="gpu", crop=crop_size, output_dtype=types.FLOAT, mean = [0.45, 0.45, 0.45], std = [0.225, 0.225, 0.225], output_layout = "CFHW")

    def define_graph(self):
        input = self.reader(name="Reader")
        crop_pos_x = self.uniform()
        crop_pos_y = self.uniform()
        is_flipped = self.coin()
        output = self.cropmirrornorm(input[0], crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y, mirror=is_flipped)
#        cropped = self.crop(input[0], crop_pos_x=crop_pos_x, crop_pos_y=crop_pos_y)
#        flipped = self.flip(cropped, horizontal=is_flipped)
#        output = self.transpose(flipped)
        # Change what you want from the dataloader.
        # input[1]: label, input[2]: starting frame number indexed from zero
        return output, input[1], input[2], crop_pos_x, crop_pos_y, is_flipped

class DALILoader():
    def __init__(self, batch_size, file_list, uid2label, sequence_length, crop_size):
        self.pipeline = VideoReaderPipeline(batch_size=batch_size,
                                            sequence_length=sequence_length,
                                            num_threads=2,
                                            device_id=0,
                                            file_list=file_list,
                                            crop_size=crop_size)
        self.pipeline.build()

        self.uid2label = uid2label
        self.epoch_size = self.pipeline.epoch_size("Reader")
        self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline,
                                                         ["data", "uid", "frame_num", "crop_pos_x", "crop_pos_y", "is_flipped"],
                                                         self.epoch_size,
                                                         auto_reset=True)
    def __len__(self):
        return int(self.epoch_size)
    def __iter__(self):
        return self
    def __next__(self):
        batch = self.dali_iterator.__next__()
        uids = batch[0]['uid'].to('cpu').data.numpy()
        batch[0]['label'] = torch.from_numpy(np.fromiter((self.uid2label[int(uid)] for uid in uids), int).reshape(uids.shape)).int().to(batch[0]['uid'].device)
        return batch[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--file_list', type=str, default='file_list.txt',
                        help='DALI file_list for VideoReader')
    parser.add_argument('--frames', type=int, default = 16,
                        help='num frames in input sequence')
    parser.add_argument('--crop_size', type=int, nargs=2, default=[224, 224],
                        help='[height, width] for input crop')
    parser.add_argument('--batchsize', type=int, default=2,
                        help='per rank batch size')

    args = parser.parse_args()

    uid2label = {35598:100, 34186:101}
    loader = DALILoader(args.batchsize,
                args.file_list,
                uid2label,
                args.frames,
                args.crop_size)
    num_samples = len(loader)
    print("Total number of samples: %d" % num_samples)

    batch = next(loader)

    print(batch['data'].shape)
    print(batch['uid'])
    print(batch['label'])
    print(batch['frame_num'])
    print(batch['crop_pos_x'])
    print(batch['crop_pos_y'])
    print(batch['is_flipped'])
