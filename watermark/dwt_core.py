import numpy as np
from numpy.linalg import svd
import copy
import cv2
from cv2 import dct, idct
from pywt import dwt2, idwt2
import warnings


class CommonPool(object):
    def map(self, func, args):
        return list(map(func, args))


class AutoPool(object):
    def __init__(self, mode, processes):

        if mode == 'multiprocessing' and sys.platform == 'win32':
            warnings.warn('multiprocessing not support in windows, turning to multithreading')
            mode = 'multithreading'

        self.mode = mode
        self.processes = processes

        if mode == 'vectorization':
            pass
        elif mode == 'cached':
            pass
        elif mode == 'multithreading':
            from multiprocessing.dummy import Pool as ThreadPool
            self.pool = ThreadPool(processes=processes)
        elif mode == 'multiprocessing':
            from multiprocessing import Pool
            self.pool = Pool(processes=processes)
        else:  # common
            self.pool = CommonPool()

    def map(self, func, args):
        return self.pool.map(func, args)
    
class WaterMarkCore:
    def __init__(self, password_img=1, mode='common', processes=None):
        self.block_shape = np.array([10, 10])
        self.password_img = password_img
        self.d1 = 36

        # init data
        self.img, self.img_YUV = None, None  
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3 
        self.ca_block = [np.array([])] * 3 
        self.ca_part = [np.array([])] * 3

        self.wm_size, self.block_num = 0, 0 
        self.pool = AutoPool(mode=mode, processes=processes)

        self.alpha = None  
        self.target_channel=0

    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size < self.block_num, IndexError(
            'You can use {}kb information as max. It is overflow {}kb information.'.format(self.block_num / 1000, self.wm_size / 1000))
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img_arr(self, img):
        self.alpha = None
        if img.shape[2] == 4:
            if img[:, :, 3].min() < 255:
                self.alpha = img[:, :, 3]
                img = img[:, :, :3]

        self.img = img.astype(np.float32)
        self.img_shape = self.img.shape[:2]

        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # construct 4 dimention block
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_wm(self, wm_bit):
        self.wm_bit = wm_bit
        self.wm_size = wm_bit.size


    def block_add_wm_fast(self, arg):
        block, i = arg
        wm_1 = self.wm_bit[i % self.wm_size]

        u, s, v = svd(dct(block))
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1

        return idct(np.dot(u, np.dot(np.diag(s), v)))

    def embed(self):
        self.init_block_index()

        # Process only the target channel
        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3
        tmp = self.pool.map(self.block_add_wm_fast,
                            [(self.ca_block[self.target_channel][self.block_index[i]], i)
                            for i in range(self.block_num)])

        for i in range(self.block_num):
            self.ca_block[self.target_channel][self.block_index[i]] = tmp[i]

        # Reassemble the processed blocks
        self.ca_part[self.target_channel] = np.concatenate(np.concatenate(self.ca_block[self.target_channel], 1), 1)
        embed_ca[self.target_channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[self.target_channel]
        embed_YUV[self.target_channel] = idwt2((embed_ca[self.target_channel], self.hvd[self.target_channel]), "haar")

        # Keep the other channels unchanged
        for channel in range(3):
            if channel != self.target_channel:
                embed_YUV[channel] = idwt2((self.ca[channel], self.hvd[channel]), "haar")
    
        # Combine the channels back into a single YUV image
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]

        # Convert to BGR color space and clip values
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

        # Add alpha channel if needed
        if self.alpha is not None:
            embed_img = cv2.merge([embed_img.astype(np.uint8), self.alpha])

        # Return the final image
        return embed_img


    def block_get_wm_fast(self, args):
        block = args
        u, s, v = svd(dct(block))
        wm = (s[0] % self.d1 > self.d1 / 2) * 1

        return wm

    def extract_raw(self, img):
        # Read the input image and initialize block indices
        self.read_img_arr(img=img)
        self.init_block_index()

        # Array to store the extracted watermark bits from the target channel
        wm_block_bit = np.zeros(shape=(self.block_num,))

        # Extract watermark bits from the target channel
        wm_block_bit[:] = self.pool.map(
            self.block_get_wm_fast,
            [
                (self.ca_block[self.target_channel][self.block_index[i]])
                for i in range(self.block_num)
            ],
        )

        return wm_block_bit

    def extract_avg(self, wm_block_bit):
        wm_avg = np.zeros(shape=self.wm_size)
        for i in range(self.wm_size):
            wm_avg[i] = wm_block_bit[i::self.wm_size].mean()
        return wm_avg

    def extract_with_kmeans(self, img, wm_shape):
        self.wm_size = np.array(wm_shape).prod()
        wm_block_bit = self.extract_raw(img=img)
        wm_avg = self.extract_avg(wm_block_bit)

        return one_dim_kmeans(wm_avg)


def one_dim_kmeans(inputs):
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()] 
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()]
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol: 
            threshold = (center[0] + center[1]) / 2
            break

    is_class01 = inputs > threshold
    return is_class01
