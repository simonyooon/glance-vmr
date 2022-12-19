import os
import sys

import numpy as np

import torch
from torch.autograd import Variable

from os.path import join
from glob import glob

import skimage.io as io
from skimage.transform import resize

from C3D_model import C3D
import cv2

import multiprocessing

from torchvision.models.feature_extraction import get_graph_node_names

# net = C3D()
# net.load_state_dict(torch.load('./c3d.pickle'))
# net.cuda()
# net.eval()


def extract_frames(video_fp, save_to_file=False, output_frames_folder=None, image_extension=None):
    """
    Function to extract frames from a given video.

    Parameters
    ----------
    :param video_fp: str
        Video file path.
    :param save_to_file: bool
        Whether to save the frame images locally.
    :param output_frames_folder: str
        Path to save output frame images.
    :param image_extension: str
        Output image file extension ('.png').

    :return: np.array
        Array of frames.
        shape = (FrameCount, height, width, channels).
    """
    start_frame = 0  # by default, the start frame of each video is the 1st frame
    cap = cv2.VideoCapture(video_fp)

    if not cap.isOpened():
        print(f'[Error] video={video_fp} can not be opened.')
        sys.exit(-6)

    # move to the start_frame
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, start_frame)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))

    if save_to_file:
        if os.path.isdir(output_frames_folder):
            print(
                f'[Warning] Given output directory \'{output_frames_folder}\' already exists. '
                f'This operation will overwrite the current directory')
        else:
            os.makedirs(output_frames_folder)

    buf = np.empty((
        frame_count,
        frame_height,
        frame_width,
        3), np.dtype('uint8'))

    for fc in range(frame_count):
        frame_num = fc + start_frame
        ret, buf[fc] = cap.read()
        if not ret:
            print(' [Error] Frame extraction was not successful')
            sys.exit(-7)
        if save_to_file:
            frame_img_name = output_frames_folder + 'frame_' + str(fc) + image_extension
            cv2.imwrite(frame_img_name, buf[fc])
    cap.release()
    videoArray = buf

    # print(videoArray.shape)
    # print(f"DURATION: {frameCount / videoFPS}")
    return videoArray


def frames_to_tensor(frames, verbose=False):
    """
    Loads a video to be fed to C3D for classification.

    Parameters
    ----------
    :param frames: np.array
        Array of frames with shape (FrameCount, height, width, channels).
    :param verbose: bool
        if True, shows the unrolled video (default is False).

    Returns
    -------
    Tensor
        a pytorch batch (n, ch, fc, h, w).
    """

    frameCount = frames.shape[0]
    frameWidth = frames.shape[1]
    frameHeight = frames.shape[2]

    # for some reason, our tutorial crop every frame to a 112 x 112 square
    frame = np.array([resize(fr, output_shape=(112, 200), preserve_range=True) for fr in frames])
    frame = frame[:, :, 44:44 + 112, :]  # crop centrally

    # visualize unrolled video
    if verbose:
        clip_img = np.reshape(frame.transpose(1, 0, 2, 3), (112, frameCount * 112, 3))
        io.imshow(clip_img.astype(np.uint8))
        io.show()

    frame = frame.transpose(3, 0, 1, 2)  # ch, fc, h, w
    frame = np.expand_dims(frame, axis=0)  # batch axis
    frame = np.float32(frame)

    return torch.from_numpy(frame)


def video_to_tensor(video_fp):
    """
    Given file path a video file, extract all the frames from the video and put them into Torch tensor.

    Parameters
    ----------
    :param video_fp: str
        File path of a folder that contains all the video files.
    :return: torch.Tensor
        a pytorch batch (n, ch, fc, h, w).
    """
    frames = extract_frames(video_fp=video_fp)
    tensor = frames_to_tensor(frames)

    return tensor


def extract_fc6(video_tensor):
    X = Variable(video_tensor)
    X = X.cuda()

    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    net = C3D()
    net.load_state_dict(torch.load('./c3d.pickle'))
    net.cuda()
    net.eval()
    net.fc6.register_forward_hook(get_activation('fc6'))

    prediction = net(X)
    return activation['fc6']


def main():
    """
    Main function.
    """
    # ungeneralized 
    video_FP_all = './data/Charades/Charades_v1'
    video_FP = './data/Charades/test1'

    # parallelization
    pool1 = multiprocessing.Pool()

    videos = sorted(glob(join(video_FP, '*.mp4')))
    print('Found ' + str(len(videos)) + ' videos at path ' + video_FP)
    try:
        # parallelize the video_to_tensor process
        video_tensors = list(pool1.map(video_to_tensor, videos))
    finally:
        pool1.close()
    print('Converted ' + str(len(video_tensors)) + ' videos to tensors')

    video_features = list(map(extract_fc6, video_tensors))

    print(video_features)
    print(video_features[0].size())


# entry point
if __name__ == '__main__':
    main()
