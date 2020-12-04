from torch import nn
from face_interpolator.models.vq_vae.res_block import ResBlock
from face_interpolator.utils.conv_transpose_2d import ConvTranspose2d


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super(Decoder, self).__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    ConvTranspose2d(nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1), output_size=(109,89)),
                    nn.ReLU(inplace=True),
                    ConvTranspose2d(nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1), output_size=(218, 178))
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)