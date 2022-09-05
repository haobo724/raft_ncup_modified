import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        # radius =4 上下左右九个为边界取样，一共81个pixel被取出来，然后拼成一条，比如输入是，[12288, 1, 96, 128]，也就是HW为96 128
        # 用[12288, 9, 9, 2] 去取样，结果输出为[12288, 1, 9, 9]，相当于 每一点（12288）有一个9*9的feature vector，然后把这个输出按照BHW重新排列得出[1, 96, 128, 81]大小
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            print('corr in pyramid:',corr.size())
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            print('corr after bilinear_sampler:',corr.size())

            corr = corr.view(batch, h1, w1, -1)
            print('corr after view:',corr.size())

            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        print('corr_fn',out.permute(0, 3, 1, 2).size())

        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        # print('after mul',corr.shape)

        # print('after transpose',fmap1.transpose(1, 2).shape)
        # print('after mul',corr.shape)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

if __name__ == '__main__':

    frame1 = torch.rand(1,3,32,32)
    frame2 = torch.rand(1,4,32,32)
    corr_example = CorrBlock(frame1,frame2)
    cord = torch.rand(5,2,32,32)
    result = corr_example(cord)
    # print(result.shape)

