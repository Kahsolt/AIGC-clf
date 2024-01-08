#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/06 

from models.utils import *

transform = T.Compose([
    VT.ToTensor(),
])


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    assert down_block_type == 'DownEncoderBlock2D'
    return DownEncoderBlock2D(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        add_downsample=add_downsample,
        resnet_eps=resnet_eps,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
        downsample_padding=downsample_padding,
    )


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    assert up_block_type == "UpDecoderBlock2D"
    return UpDecoderBlock2D(
        num_layers=num_layers,
        in_channels=in_channels,
        out_channels=out_channels,
        add_upsample=add_upsample,
        resnet_eps=resnet_eps,
        resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups,
    )


def upsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = Tensor(kernel, dtype=ms.float32)
    if kernel.ndim == 1:
        kernel = F.outer(kernel, kernel)
    kernel /= F.sum(kernel)

    kernel = kernel * (gain * (factor**2))
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        kernel.to(device=hidden_states.device),
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output


def downsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = Tensor(kernel, dtype=ms.float32)
    if kernel.ndim == 1:
        kernel = F.outer(kernel, kernel)
    kernel /= F.sum(kernel)

    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states, 
        kernel.to(device=hidden_states.device), 
        down=factor, 
        pad=((pad_value + 1) // 2, pad_value // 2)
    )
    return output


def upfirdn2d_native(tensor, kernel, up=1, down=1, pad=(0, 0)):
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]

    _, channel, in_h, in_w = tensor.shape
    tensor = tensor.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = tensor.shape
    kernel_h, kernel_w = kernel.shape

    out = tensor.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out.to(tensor.device)  # Move back to mps if necessary
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = F.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


class Upsample2D(nn.Cell):

    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            conv = nn.Conv2dTranspose(channels, self.out_channels, 4, 2, 1, has_bias=True)
        elif use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, pad_mode='pad', padding=1, has_bias=True)
        self.conv = conv

    def construct(self, hidden_states, output_size=None):
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == ms.bfloat16:
            hidden_states = hidden_states.to(ms.float32)

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            B, C, H, W = hidden_states.shape
            output_size = (H * 2, W * 2)
        hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == ms.bfloat16:
            hidden_states = hidden_states.to(dtype)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if self.use_conv:
            hidden_states = self.conv(hidden_states)
        return hidden_states


class Downsample2D(nn.Cell):

    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()

        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2d(self.channels, self.out_channels, 3, stride=stride, pad_mode='pad', padding=padding, has_bias=True)
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def construct(self, hidden_states):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = F.pad(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states


class ResnetBlock2D(nn.Cell):

    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        kernel=None,
        output_scale_factor=1.0,
        use_in_shortcut=None,
        up=False,
        down=False,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)

        if temb_channels is not None:
            self.time_emb_proj = nn.Dense(temb_channels, out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = nn.Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(F.avg_pool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(in_channels, use_conv=False, padding=1, name="op")

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=True)

    def construct(self, input_tensor, temb):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)

        hidden_states = self.conv1(hidden_states)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor


class UpDecoderBlock2D(nn.Cell):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.CellList(resnets)

        if add_upsample:
            self.upsamplers = nn.CellList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

    def construct(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class DownEncoderBlock2D(nn.Cell):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
        self.resnets = nn.CellList(resnets)

        if add_downsample:
            self.downsamplers = nn.CellList([
                Downsample2D(in_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op")
            ])
        else:
            self.downsamplers = None

    def construct(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class AttentionBlock(nn.Cell):

    def __init__(
        self,
        channels: int,
        num_head_channels: Optional[int] = None,
        norm_num_groups: int = 32,
        rescale_output_factor: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = channels // num_head_channels if num_head_channels is not None else 1
        self.num_head_size = num_head_channels
        self.group_norm = nn.GroupNorm(num_channels=channels, num_groups=norm_num_groups, eps=eps, affine=True)

        # define q,k,v as linear layers
        self.query = nn.Dense(channels, channels)
        self.key = nn.Dense(channels, channels)
        self.value = nn.Dense(channels, channels)

        self.rescale_output_factor = rescale_output_factor
        self.proj_attn = nn.Dense(channels, channels, 1)

    def transpose_for_scores(self, projection: Tensor) -> Tensor:
        new_projection_shape = projection.shape[:-1] + (self.num_heads, -1)
        # move heads to 2nd position (B, T, H * D) -> (B, T, H, D) -> (B, H, T, D)
        new_projection = projection.view(new_projection_shape).permute(0, 2, 1, 3)
        return new_projection

    def construct(self, hidden_states):
        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)
        hidden_states = hidden_states.view(batch, channel, height * width).swapaxes(1, 2)

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        # transpose
        query_states = self.transpose_for_scores(query_proj)
        key_states = self.transpose_for_scores(key_proj)
        value_states = self.transpose_for_scores(value_proj)

        # get scores
        scale = 1 / math.sqrt(math.sqrt(self.channels / self.num_heads))
        attention_scores = F.matmul(query_states * scale, key_states.swapaxes(-1, -2) * scale)  # TODO: use baddmm
        attention_probs = F.softmax(attention_scores.float(), axis=-1).type(attention_scores.dtype)

        # compute attention output
        hidden_states = F.matmul(attention_probs, value_states)
        hidden_states = hidden_states.permute(0, 2, 1, 3)   # .contiguous()
        new_hidden_states_shape = hidden_states.shape[:-2] + (self.channels,)
        hidden_states = hidden_states.view(new_hidden_states_shape)

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states


class UNetMidBlock2D(nn.Cell):

    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        attention_type="default",
        output_scale_factor=1.0,
        **kwargs,
    ):
        super().__init__()

        self.attention_type = attention_type
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock2D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        for _ in range(num_layers):
            attentions.append(
                AttentionBlock(
                    in_channels,
                    num_head_channels=attn_num_head_channels,
                    rescale_output_factor=output_scale_factor,
                    eps=resnet_eps,
                    norm_num_groups=resnet_groups,
                )
            )
            resnets.append(
                ResnetBlock2D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.CellList(attentions)
        self.resnets = nn.CellList(resnets)

    def construct(self, hidden_states, temb=None, encoder_states=None):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if self.attention_type == "default":
                hidden_states = attn(hidden_states)
            else:
                hidden_states = attn(hidden_states, encoder_states)
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


class Encoder(nn.Cell):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
        double_z=True,
    ):
        super().__init__()

        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.mid_block = None
        self.down_blocks = nn.CellList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, pad_mode='pad', padding=1, has_bias=True)

    def construct(self, x):
        sample = x
        sample = self.conv_in(sample)
        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)
        # middle
        sample = self.mid_block(sample)
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class Decoder(nn.Cell):

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(64,),
        layers_per_block=2,
        norm_num_groups=32,
        act_fn="silu",
    ):
        super().__init__()

        self.layers_per_block = layers_per_block
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=True)
        self.mid_block = None
        self.up_blocks = nn.CellList([])

        # mid
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attn_num_head_channels=None,
            resnet_groups=norm_num_groups,
            temb_channels=None,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            up_block = get_up_block(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attn_num_head_channels=None,
                temb_channels=None,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, pad_mode='pad', padding=1, has_bias=True)

    def construct(self, z):
        sample = z
        sample = self.conv_in(sample)
        # middle
        sample = self.mid_block(sample)
        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)
        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class DiagonalGaussianDistribution:

    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = F.chunk(parameters, 2, axis=1)
        self.logvar = F.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = F.exp(0.5 * self.logvar)
        self.var = F.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = F.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)

    def sample(self) -> Tensor:
        device = self.parameters.device
        sample_device = "cpu" if device.type == "mps" else device
        sample = F.randn(self.mean.shape).to(device=sample_device)
        # make sure sample is on the same device as the parameters and has same dtype
        sample = sample.to(device=device, dtype=self.parameters.dtype)
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return F.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * F.sum(F.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * F.sum(
                    F.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * F.sum(logtwopi + self.logvar + F.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Cell):

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        **kwargs,
    ):
        super().__init__()

        # pass init params to Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
        )

        # pass init params to Decoder
        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1, has_bias=True)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1, has_bias=True)

    def encode(self, x: Tensor) -> DiagonalGaussianDistribution:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def construct(
        self,
        sample: Tensor,
        sample_posterior: bool = False,
        generator: Optional[Generator] = None,
    ) -> Tensor:
        x = sample
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec


def param_dict_name_mapping(kv:Dict[str, Parameter]) -> Dict[str, Parameter]:
    new_kv = {}
    for k in kv.keys():
        new_k = k
        # GroupNorm
        if k.endswith('.conv_norm_out.weight'): new_k = k.replace('.conv_norm_out.weight', '.conv_norm_out.gamma')
        if k.endswith('.conv_norm_out.bias'):   new_k = k.replace('.conv_norm_out.bias',   '.conv_norm_out.beta')
        if k.endswith('.group_norm.weight'):    new_k = k.replace('.group_norm.weight',    '.group_norm.gamma')
        if k.endswith('.group_norm.bias'):      new_k = k.replace('.group_norm.bias',      '.group_norm.beta')
        if k.endswith('.norm1.weight'):         new_k = k.replace('.norm1.weight',         '.norm1.gamma')
        if k.endswith('.norm1.bias'):           new_k = k.replace('.norm1.bias',           '.norm1.beta')
        if k.endswith('.norm2.weight'):         new_k = k.replace('.norm2.weight',         '.norm2.gamma')
        if k.endswith('.norm2.bias'):           new_k = k.replace('.norm2.bias',           '.norm2.beta')
        new_kv[new_k] = kv[k]
    return new_kv


def infer_autoencoder_kl(model:AutoencoderKL, img:PILImage) -> Tuple[Tensor, Tensor]:
    def pad(x:Tensor, opt_C:int=8) -> Tuple[Tensor, Tuple[int]]:
        C, H, W = x.shape
        H_pad = math.ceil(H / opt_C) * opt_C - H
        W_pad = math.ceil(W / opt_C) * opt_C - W
        pL, pR, pT, pB = W_pad//2, W_pad-W_pad//2, H_pad//2, H_pad-H_pad//2
        if min(pL, pR, pT, pB) > 0:
            x = F.pad(x, padding=(pL, pR, pT, pB), mode='reflect')
        return x, (pL, pR, pT, pB)
    def unpad(x:Tensor, pads:Tuple[int]) -> Tensor:
        pL, pR, pT, pB = pads
        if min(pL, pR, pT, pB) > 0:
            sH = slice(pT, -pB if pB else None)
            sW = slice(pL, -pR if pR else None)
            x = x[:, sH, sW]
        return x

    X = transform(img)[0]   # do not know why this returns a ndarray tuple
    X = Tensor.from_numpy(X).astype(ms.float32)
    X_pad, pads = pad(X)
    X_pad_hat = model(X_pad.unsqueeze(0)).squeeze(0)
    X_hat = unpad(X_pad_hat, pads)
    return X, X_hat


def get_app(app_name:str) -> AutoencoderKL:
    HF_PATH = Path(__file__).parent.parent / 'huggingface'
    APP_PATH = HF_PATH / app_name
    CONFIG_FILE = APP_PATH / 'model.json'
    WEIGHT_FILE = APP_PATH / 'model.npz'

    with open(CONFIG_FILE) as fh:
        cfg = json.load(fh)

    model = AutoencoderKL(**cfg)
    model = model.set_train(False)
    param_dict = load_npz_as_param_dict(WEIGHT_FILE)
    param_dict = param_dict_name_mapping(param_dict)
    if 'ckeck keys match':
        model_keys = set(model.parameters_dict().keys())
        ckpt_keys = set(param_dict.keys())
        missing_keys = model_keys - ckpt_keys
        redundant_keys = ckpt_keys - model_keys
        if redundant_keys or missing_keys:
            print('redundant keys:', redundant_keys)
            print('missing keys:', missing_keys)
            breakpoint()
    ms.load_param_into_net(model, param_dict)
    return model


def get_sd_vae_ft_ema() -> AutoencoderKL:
    return get_app('stabilityai#sd-vae-ft-ema')


def get_sd_vae_ft_mse() -> AutoencoderKL:
    return get_app('stabilityai#sd-vae-ft-mse')


if __name__ == '__main__':
    model = get_sd_vae_ft_ema()
    print(model)
    X = F.ones([1, 3, 224, 224]) * 0.5
    X_hat = model(X)
    print(X_hat.shape)
    err = (X_hat - X).abs().mean().item()
    print(f'>> err: {err}')
