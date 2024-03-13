from typing import Dict, Union, Tuple, Callable, Optional
from torch import Tensor, tensor
import torch
import torch.nn as nn
import numpy as np
from kornia.filters.filter import filter3d

class LibraryTransforms:
    pass
    '''
    Monai - look for dict transforms for affines etc! - please verify that txf is all torch for speedy augs
        RandomDepthicalFlip3D()
        RandomHorizontalFlip3D()
        RandomVerticalFlip3D()
        RandomRotation3D()
        RandomAffine3D()
        RandomCrop3D()
    '''

    '''
    torchIO wrappers - https://torchio.readthedocs.io/transforms/transforms.html#transforms-api
        RandomBiasField()
    '''
    #

class CustomTransforms:
    '''
        Custom transform methods
        implements randdom erase and wrapped dict implementations
        implements wrapped random affine based class and extensions
    '''
    @staticmethod
    def denoise(images: Tensor, 
                masks: Tensor)->Tensor:
        '''
            Skull stripping with mask
            @param: images: Tensor; batches of images with shape [batch,channel,Z,Y,X]
            @param: masks: Tensor; brain mask of shape [batch, 1 ,Z,Y,X]
            @return Tensor ~ transformed images
        '''
        images = images.mul(masks)
        return images

    @staticmethod
    def standardize(images: Tensor, 
                    masks: Tensor, 
                    eps: float=1e-12)->Tensor:
        '''
            Standardize to 0 mean, 1 std. dev. in brain (masked/skull stripped)
            @param: images: Tensor; batches of images with shape [batch,channel,Z,Y,X]
            @param: masks: Tensor; brain mask of shape [batch, 1 ,Z,Y,X]
            @return Tensor ~ transformed image
        '''
        images = images.mul(masks)
        N = masks.sum(dim=(2,3,4), keepdims=True)
        means = images.sum(dim=(2,3,4), keepdims=True) / N
        stds = torch.sqrt(images.pow(2).sum(dim=(2,3,4), keepdims=True) / N - means.pow(2) + eps)
        images = images.sub(means)
        images = images.div(stds)
        images = images.mul(masks)
        return images

    @staticmethod
    def bind(images: Tensor, 
             masks: Tensor, 
             eps: float=1e-12)->Tensor:

        '''
            Normalizing to 0 min, 1 max, in brain (masked/skull stripped), 3 std. dev. cutoff
            @param: images: Tensor; batches of images with shape [batch,channel,Z,Y,X]
            @param: masks: Tensor; brain mask of shape [batch, 1 ,Z,Y,X]
        '''

        images = images.mul(masks)
        b,c,_,_,_ = images.size()
        N = masks.sum(dim=(2,3,4), keepdims=True) # across Z,Y,X dimensions
        means = images.sum(dim=(2,3,4), keepdims=True) / N
        stds = torch.sqrt(images.pow(2).sum(dim=(2,3,4), keepdims=True) / N - means.pow(2) + eps)
        images = images.mul(masks) # background clean up
        images = torch.maximum(images, means-3*stds)
        images = torch.minimum(images, means+3*stds)
        images = images.add( (1-masks)*1e10) # set background to high number to get only min. of brain
        images = images.sub(torch.min(images.view(b,c,-1), dim=2)[0].view(b,c,1,1,1))
        images = images.mul(masks) # background clean up
        images = images.div(torch.max(images.view(b,c,-1), dim=2)[0].view(b,c,1,1,1))
        images = images.mul(masks) # background clean up

        return images

    @staticmethod
    def std_bind(images: Tensor, 
                 masks: Tensor)->Tensor:

        '''
            On a standardized image: normalize to 0 min, 1 max, in brain (masked/skull stripped), 3 std. dev. cutoff
            @param: images: Tensor; batches of images with shape [batch,channel,Z,Y,X]
            @param: masks: Tensor; brain mask of shape [batch, 1 ,Z,Y,X]
            @return Tensor ~ transformed images
        '''

        masks = masks.squeeze(1)
        images = images.transpose(1,0)
        c, b, _, _, _ = images.size()
        images = torch.clip(images, -3, 3)
        images = images.sub(torch.min(images.view(c,b,-1), dim=2)[0].view(c,b,1,1,1))
        images = images.div(torch.max(images.view(c,b,-1), dim=2)[0].view(c,b,1,1,1))
        images = images.transpose(1,0)
        images = images.mul(masks.unsqueeze(1)) # denoise/mask background

        return images

    class _RandomSampling3D():
        ''' base class for any patch sampling transformation '''

        def __init__(self,
                    min_dim: Union[int, Tuple[int, int, int]],
                    max_dim: Tuple[int, int, int],
                    same_on_batch=False,
                    device=torch.device('cpu')):

            self.min_dim = min_dim if type(min_dim) == list else 3*[min_dim]
            self.max_dim = max_dim if type(max_dim) == list else 3*[max_dim]
            self.same_on_batch = same_on_batch
            self.device = device

        def sample(self, dim_b:int, dim_z:int, dim_y: int, dim_x: int)->Tuple:
            if self.same_on_batch: size = (1,)
            else: size = (dim_b,)

            # sample random location within image
            pz = torch.randint(low=0, high=dim_z-self.max_dim[0], size=size, device=self.device)
            py = torch.randint(low=0, high=dim_y-self.max_dim[1], size=size, device=self.device)
            px = torch.randint(low=0, high=dim_x-self.max_dim[2], size=size, device=self.device)

            # sample region
            dz = torch.randint(low=self.min_dim[0], high=self.max_dim[0]+1, size=size, device=self.device)
            dy = torch.randint(low=self.min_dim[1], high=self.max_dim[1]+1, size=size, device=self.device)
            dx = torch.randint(low=self.min_dim[2], high=self.max_dim[2]+1, size=size, device=self.device)

            return (pz, py, px, dz, dy, dx)

    class RandomErasing3D(_RandomSampling3D):
        '''
            Perform random erasing on a set o input brain volumes
            pipeline implementation generalized for 3d volumes

            @method: callable
            @param: p: probability of occurance
            @param: min_dim: int or 3list for minimum axis lengths
            @param: max_dim: int 3list for maximum axis lengths
            @param: value: replacement value default 0
            @param: same_on_batch: same erasing over batch dim, default false
            @return Tensor ~ transformed image
        '''

        def __init__(self,
                    min_dim: Union[int, Tuple[int, int, int]],
                    max_dim: Tuple[int, int, int],
                    value: int=0,
                    p: float=1.0,
                    same_on_batch=False,
                    device = torch.device('cpu')):
            self.value = value
            self.p = p
            self.same_on_batch = same_on_batch
            self.device = device
            super().__init__(min_dim, max_dim, same_on_batch, device)

        def __call__(self, imgs: Tensor)->Tensor:

            # reject txf
            if torch.rand(1, device=self.device) > self.p: return imgs

            # verify mode
            dim_b, _, dim_z, dim_y, dim_x = imgs.size()

            # samples
            pz, py, px, dz, dy, dx = self.sample(dim_b, dim_z, dim_y, dim_x)

            # erase valid volume over sequences
            if self.same_on_batch: imgs[:,:,pz:pz+dz, py:py+dy, px:px+dx] = self.value
            else:
                for b in range(dim_b):
                    imgs[b,:,pz[b]:pz[b]+dz[b], py[b]:py[b]+dy[b], px[b]:px[b]+dx[b]] = self.value

            return imgs

    class WrappedRandomErasing3D(_RandomSampling3D):

        ''' input mapping as list imaging keys. See RandomErasing3D '''

        def __init__(self,
                    input_mapping: list,
                    min_dim: Union[int, Tuple[int, int, int]],
                    max_dim: Tuple[int, int, int],
                    value: int=0,
                    p: float=1.0,
                    same_on_batch=False,
                    device = torch.device('cpu')):

            self.mapping = input_mapping
            self.value = value
            self.p = p
            super().__init__(min_dim, max_dim, same_on_batch, device)

        def __call__(self, info: Dict)->Dict:
            if torch.rand(1, device=self.device) > self.p: return info
            dim_b, _, dim_z, dim_y, dim_x = info[self.mapping[0]].size()
            pz, py, px, dz, dy, dx = self.sample(dim_b, dim_z, dim_y, dim_x)
            for key in self.mapping:
                imgs = info[key].clone()
                if self.same_on_batch:
                    imgs[:,:,pz:pz+dz, py:py+dy, px:px+dx] = self.value
                else:
                    for b in range(dim_b):
                        imgs[b,:,pz[b]:pz[b]+dz[b], py[b]:py[b]+dy[b], px[b]:px[b]+dx[b]] = self.value
                info[key] = imgs
            
            return info

    class WrappedRandomCropping3D(_RandomSampling3D):
        '''
            Perform random cropping on a set of input brain volumes
            pipeline implementation generalized for 3d volumes
            @method: callable
            @param: min_dim: int or 3list for minimum axis lengths
            @param: max_dim: int 3list for maximum axis lengths
            @param: same_on_batch: same erasing over batch dim, default false
            @return Tensor ~ transformed image
        '''

        def __init__(self,
                    input_mapping: list,
                    size: Tuple[int, int, int],
                    same_on_batch=False,
                    device = torch.device('cpu')):
            self.mapping = input_mapping
            self.same_on_batch = same_on_batch
            self.device = device

            min_dim = size
            super().__init__(min_dim, min_dim, same_on_batch, device)

        def __call__(self, info: Dict)->Dict:
            dim_b, _, dim_z, dim_y, dim_x = info[self.mapping[0]].size()
            pz, py, px, dz, dy, dx = self.sample(dim_b, dim_z, dim_y, dim_x)
            for key in self.mapping:
                imgs = info[key].clone()
                if self.same_on_batch:
                    imgs = imgs[:,:,pz:pz+dz, py:py+dy, px:px+dx]
                else:
                    imgs = torch.stack([imgs[b,:,pz[b]:pz[b]+dz[b], py[b]:py[b]+dy[b], px[b]:px[b]+dx[b]]
                        for b in range(dim_b)])
                info[key] = imgs
            
            return info
        
    class RandomGaussianBlur3D():
        '''
            Randomly sample a gaussian kernel given a size and sigma and apply it to a set of images
            Blurs all images along the channel dimension
            @method: callable
            @param: kernel_size: (K,K,K) kernel generated, must be odd
            @param: sigma: tuple of (low, high) ranges sampled uniformly
            @param: p: prob that the transform is applied 
            @param: device: torch device which stores kernel

            Pre-allocates a standard normal kernel at init time and applied the sampled sigma on call 
        '''
        
        def __init__(self, 
                     kernel_size: int, 
                     sigma: Tuple[float, float], 
                     p: float, 
                     device=torch.device('cpu')):
            assert kernel_size % 2 != 0, "Kernel size must be odd"
            self.kernel_size = kernel_size
            self.sigma = sigma
            self.p = p
            self.device = device
            self.base = self._generate_base()
            self.const = torch.pow(torch.tensor([2*np.pi], device=device), 3/2)

        def _generate_base(self)->Tensor:
            ''' generate unnormalized unit normal kernel '''
            base = torch.empty((self.kernel_size, self.kernel_size, self.kernel_size), device=self.device)
            m = self.kernel_size // 2
            for i in range(self.kernel_size):
                for j in range(self.kernel_size):
                    for k in range(self.kernel_size):
                        base[i,j,k] = (i - m)**2 + (j - m)**2 + (k - m)**2
            base = base.float()
            base = torch.exp(-base/2)
            return base

        def _sample(self)->Tensor:
            low, high = self.sigma
            return low + torch.rand(1, device=self.device) * (high - low)
        
        def __call__(self, x: Tensor)->Tensor:
            if torch.rand(1, device=self.device) < self.p:
                sigma = self._sample()
                kernel = (torch.pow(self.base, 1.0/torch.pow(sigma, 2)) / (self.const * torch.pow(sigma, 3))).unsqueeze(0)
                x = filter3d(x, kernel)
            return x
        
    class RandomGaussianNoise():
        '''
            Randomly sample noise from a gaussian distribution with target mean and std 
            Apply uniquely to each channel
            @method: callable: apply random noise
            @param: mean: float mean across all channels
            @param: std: float std across all channels
            @param: p: probability of applying the transform
            @param: device to generate noise on
        '''
        
        def __init__(self, mean: float, 
                     std: float, 
                     p: float, 
                     device=torch.device('cpu')):
            self.mean = mean
            self.std = std
            self.p = p
            self.device = device

        def __call__(self, x: Tensor)->Tensor:
            if torch.rand(1, device=self.device) < self.p:
                mean = torch.empty_like(x, device=self.device).fill_(self.mean)
                std = torch.empty_like(x, device=self.device).fill_(self.std)
                x += torch.normal(mean, std)
            return x

    class RandomAdjustContrast3D():
        '''
            Implements a pytorch gpu enabled color jitter
            Broadcastable to each channel
            Option to jitter only in the mask
            Please consider applying the mask downstream to this operation
            @method: callable: apply transform
            @param: channels
            @param: gamma
        '''
        
        def __init__(self, 
                     gamma: Tuple[float, float], 
                     p: float, 
                     same_on_channel=False, 
                     device=torch.device('cpu'), 
                     eps=1e-8):
            self.gamma_min, self.gamma_max = gamma
            self.same_on_channel = same_on_channel
            self.p = p
            self.device = device
            self.eps = eps

        def _sample_gamma(self, channels: Tuple)->Tensor:
            rand = torch.rand(channels, device=self.device)
            return self.gamma_min + rand * (self.gamma_max - self.gamma_min)
        
        def __call__(self, x: Tensor, mask: Tensor=None)->Tensor:
            if torch.rand(1, device=self.device) < self.p:   
                x_min = x.min() if mask is None else x[mask].min()
                x_max = x.max() if mask is None else x[mask].max()
                channels = 1 if self.same_on_channel else x.shape[1]
                gamma = self._sample_gamma(channels).view(1, -1, 1, 1 ,1)
                x = torch.pow((x - x_min) / (x_max - x_min + self.eps), gamma) * (x_max - x_min) + x_min
            return x
    
    class RandomFlip():
        '''
            Nd version of random flip
            Flip items along randomly sampled dimensions
            Invertable
            
            NOTE: Nd specifies number of dimensions not including batch and channel dims.
                  e.g. for tensor (b,c,d,w,h), ndim=3 selects flips on (d,w,h) dimensions
        '''
        def __init__(self, p: float, 
                        ndim: int, 
                        return_params=False):
            self.p = p
            self.ndim = ndim
            self.return_params = return_params

        def _sample_dims(self)->Tensor:
            dims = torch.arange(self.ndim)
            samples = torch.rand(self.ndim)
            sampled_dims = dims[samples < self.p]
            return sampled_dims

        def _flip(self, item: Tensor, dims: Tensor)->Tensor:
            flip_dims = tuple(dims + item.ndim - self.ndim)
            return torch.flip(item, flip_dims)

        def __call__(self, item: Tensor)->Tensor:
            sampled_dims = self._sample_dims()
            flipped = self._flip(item, sampled_dims)
            if self.return_params:
                return flipped, sampled_dims
            else:
                return flipped

        def invert(self, item: Tensor, dims: Tensor)->Tensor:
            flipped = self._flip(item, dims)
            return flipped

    class RandomPermute():
        '''
            Nd version of random permute
            permute dims along over last set of dims
            Invertable
            
            NOTE: Nd specifies number of dimensions not including batch and channel dims.
              e.g. for tensor (b,c,d,w,h), ndim=3 does permutations on (d,w,h) dimensions
        '''
        def __init__(self, 
                        p: float, 
                        ndim: int, 
                        return_params=False):
            self.p = p
            self.ndim = ndim
            self.return_params = return_params

        def _sample_dims(self)->Tensor:
            return torch.randperm(self.ndim)

        def _permute(self, item: Tensor, dims: Tensor)->Tensor:
            base_dims = item.ndim - self.ndim
            permuted_dims = dims + base_dims
            return item.permute(*torch.arange(base_dims), *permuted_dims)

        def __call__(self, item: Tensor)->Tensor:
            sampled_dims = self._sample_dims()
            permuted = self._permute(item, sampled_dims)
            if self.return_params:
                return permuted, sampled_dims
            else:
                return permuted

        def invert(self, item: Tensor, dims: Tensor)->Tensor:
            # note: dims (for permute) is always monotonic, hence sorting
            #       in ascending order is the inverse transform
            reverse_dims = torch.argsort(dims)
            permuted = self._permute(item, reverse_dims)
            return permuted
        
    class RandomQuantizedRotate():
        '''
            Nd version of random rotation using flip permute combos
            rotate image by 90deg intervals
            Invertable
            
            NOTE: Nd specifies number of dimensions not including batch and channel dims.
                e.g. for tensor (b,c,d,w,h), ndim=3 performs random rotation on (d,w,h) dimensions
        '''
        def __init__(self, 
                     p_flip: float, 
                     p_permute: float, 
                     ndim: int, 
                     return_params=False):
            self.transforms = [CustomTransforms.RandomFlip(p_flip, ndim, True), 
                               CustomTransforms.RandomPermute(p_permute, ndim, True)]
            self.return_params = return_params

        def __call__(self, item: Tensor)->Tensor:
            order = torch.multinomial(torch.ones(2), 2) # selects order of permutation/flip transforms
            txf_params = [None]*2
            for idx in order:
                item, param = self.transforms[idx](item)
                txf_params[idx] = param
            if self.return_params:
                return item, (order, txf_params)
            else:
                return item

        def invert(self, item: Tensor, params: Tuple)->Tensor:
            order, txf_params = params
            for idx in torch.flip(order, (0,)):
                param = txf_params[idx]
                item = self.transforms[idx].invert(item, param)
            return item
