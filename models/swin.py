#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/04 

from itertools import repeat
from models.utils import *
try: from mindcv.models.swin_transformer import swin_tiny
except: pass

transform = T.Compose([
    VT.RandomResizedCrop(size=(244, 244)),
    VT.ToTensor(),
    VT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False),
])


class SwinConfig:

    model_type = "swin"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        encoder_stride=32,
        chunk_size_feed_forward=0,
        **kwargs,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.encoder_stride = encoder_stride
        # we set the hidden_size attribute in order to make Swin work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        self.chunk_size_feed_forward = chunk_size_feed_forward


def drop_path(input, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training: return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0],) + (1,) * (input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + F.rand(shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output

def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def window_partition(x:ndarray, window_size: int) -> ndarray:
    b, h, w, c = x.shape
    x = np.reshape(x, (b, h // window_size, window_size, w // window_size, window_size, c))
    windows = x.transpose(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, c)
    return windows


class WindowPartition(nn.Cell):

    def __init__(
        self,
        window_size: int,
    ) -> None:
        super(WindowPartition, self).__init__()

        self.window_size = window_size

    def construct(self, x: Tensor) -> Tensor:
        b, h, w, c = x.shape
        x = F.reshape(x, (b, h // self.window_size, self.window_size, w // self.window_size, self.window_size, c))
        x = F.transpose(x, (0, 1, 3, 2, 4, 5))
        x = F.reshape(x, (b * h * w // (self.window_size**2), self.window_size, self.window_size, c))
        return x


class WindowReverse(nn.Cell):

    def construct(
        self,
        windows: Tensor,
        window_size: int,
        h: int,
        w: int,
    ) -> Tensor:
        b = windows.shape[0] // (h * w // window_size // window_size)
        x = F.reshape(windows, (b, h // window_size, w // window_size, window_size, window_size, -1))
        x = F.transpose(x, (0, 1, 3, 2, 4, 5))
        x = F.reshape(x, (b, h, w, -1))
        return x


class Roll(nn.Cell):

    def __init__(self, shift_size: int, shift_axis: Tuple[int] = (1, 2)):
        super().__init__()

        self.shift_size = to_2tuple(shift_size)
        self.shift_axis = shift_axis

    def construct(self, x: Tensor) -> Tensor:
        x = np.roll(x.numpy(), self.shift_size, self.shift_axis)
        return Tensor.from_numpy(x)


class SwinEmbeddings(nn.Cell):

    def __init__(self, config, use_mask_token=False):
        super().__init__()

        self.patch_embeddings = SwinPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.patch_grid = self.patch_embeddings.grid_size
        self.mask_token = nn.Parameter(F.zeros(1, 1, config.embed_dim)) if use_mask_token else None

        if config.use_absolute_embeddings:
            self.position_embeddings = nn.Parameter(F.zeros(1, num_patches + 1, config.embed_dim))
        else:
            self.position_embeddings = None

        self.norm = nn.LayerNorm([config.embed_dim])
        self.dropout = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob else nn.Identity()

    def construct(
        self, pixel_values: Tensor, bool_masked_pos: Tensor = None
    ) -> Tuple[Tensor]:
        embeddings, output_dimensions = self.patch_embeddings(pixel_values)
        embeddings = self.norm(embeddings)
        batch_size, seq_len, _ = embeddings.shape

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(batch_size, seq_len, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        if self.position_embeddings is not None:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings, output_dimensions


class SwinPatchEmbeddings(nn.Cell):

    def __init__(self, config):
        super().__init__()

        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.embed_dim
        image_size = image_size if isinstance(image_size, Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, has_bias=True)

    def maybe_pad(self, pixel_values, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            pixel_values = F.pad(pixel_values, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            pixel_values = F.pad(pixel_values, pad_values)
        return pixel_values

    def construct(self, pixel_values: Tensor) -> Tuple[Tensor, Tuple[int]]:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError("Make sure that the channel dimension of the pixel values match with the one set in the configuration.")
        # pad the input to be divisible by self.patch_size, if needed
        pixel_values = self.maybe_pad(pixel_values, height, width)
        embeddings = self.projection(pixel_values)
        _, _, height, width = embeddings.shape
        output_dimensions = (height, width)
        embeddings = embeddings.flatten(start_dim=2).permute(0, 2, 1)
        return embeddings, output_dimensions


class SwinPatchMerging(nn.Cell):

    def __init__(self, input_resolution: Tuple[int], dim: int, norm_layer: nn.Cell = nn.LayerNorm):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Dense(4 * dim, 2 * dim, has_bias=False)
        self.norm = norm_layer([4 * dim])

    def maybe_pad(self, input_feature, height, width):
        should_pad = (height % 2 == 1) or (width % 2 == 1)
        if should_pad:
            pad_values = (0, 0, 0, width % 2, 0, height % 2)
            input_feature = F.pad(input_feature, pad_values)
        return input_feature

    def construct(self, input_feature: Tensor, input_dimensions: Tuple[int, int]) -> Tensor:
        height, width = input_dimensions
        # `dim` is height * width
        batch_size, dim, num_channels = input_feature.shape

        input_feature = input_feature.view(batch_size, height, width, num_channels)
        # pad input to be disible by width and height, if needed
        input_feature = self.maybe_pad(input_feature, height, width)
        # [batch_size, height/2, width/2, num_channels]
        input_feature_0 = input_feature[:, 0::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_1 = input_feature[:, 1::2, 0::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_2 = input_feature[:, 0::2, 1::2, :]
        # [batch_size, height/2, width/2, num_channels]
        input_feature_3 = input_feature[:, 1::2, 1::2, :]
        # batch_size height/2 width/2 4*num_channels
        input_feature = F.concat([input_feature_0, input_feature_1, input_feature_2, input_feature_3], -1)
        input_feature = input_feature.view(batch_size, -1, 4 * num_channels)  # batch_size height/2*width/2 4*C

        input_feature = self.norm(input_feature)
        input_feature = self.reduction(input_feature)
        return input_feature


class SwinDropPath(nn.Cell):

    def __init__(self, drop_prob: float = None):
        super().__init__()

        self.drop_prob = drop_prob

    def construct(self, hidden_states: Tensor) -> Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SwinSelfAttention(nn.Cell):

    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})")

        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, Iterable) else (window_size, window_size)
        )

        self.relative_position_bias_table = Parameter(
            F.zeros(((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads))
        )

        # get pair-wise relative position index for each token inside the window
        coords_h = F.arange(self.window_size[0])
        coords_w = F.arange(self.window_size[1])
        coords = F.stack(F.meshgrid(*[coords_h, coords_w], indexing="ij"))
        coords_flatten = F.flatten(coords, start_dim=1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)      # .contiguous(), FIXME: this causes deadloop
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.relative_position_index = Parameter(relative_coords.sum(-1))

        self.query = nn.Dense(self.all_head_size, self.all_head_size, has_bias=config.qkv_bias)
        self.key = nn.Dense(self.all_head_size, self.all_head_size, has_bias=config.qkv_bias)
        self.value = nn.Dense(self.all_head_size, self.all_head_size, has_bias=config.qkv_bias)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob) if config.attention_probs_dropout_prob else nn.Identity()

    def transpose_for_scores(self, x):
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
    ) -> Tuple[Tensor]:
        batch_size, dim, num_channels = hidden_states.shape
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = F.matmul(query_layer, key_layer.permute(0, 1, 3, 2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )

        relative_position_bias = relative_position_bias.permute(2, 0, 1)    # .contiguous()
        attention_scores = attention_scores + relative_position_bias.unsqueeze(0)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in SwinModel forward() function)
            mask_shape = attention_mask.shape[0]
            attention_scores = attention_scores.view(
                batch_size // mask_shape, mask_shape, self.num_attention_heads, dim, dim
            )
            attention_scores = attention_scores + attention_mask.unsqueeze(1).unsqueeze(0)
            attention_scores = attention_scores.view(-1, self.num_attention_heads, dim, dim)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, axis=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = F.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3)   # .contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer


class SwinSelfOutput(nn.Cell):

    def __init__(self, config, dim):
        super().__init__()

        self.dense = nn.Dense(dim, dim)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob) if config.attention_probs_dropout_prob else nn.Identity()

    def construct(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class SwinAttention(nn.Cell):

    def __init__(self, config, dim, num_heads, window_size):
        super().__init__()

        self.self = SwinSelfAttention(config, dim, num_heads, window_size)
        self.output = SwinSelfOutput(config, dim)

    def construct(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
    ) -> Tuple[Tensor]:
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class SwinIntermediate(nn.Cell):

    def __init__(self, config, dim):
        super().__init__()

        self.dense = nn.Dense(dim, int(config.mlp_ratio * dim))
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class SwinOutput(nn.Cell):

    def __init__(self, config, dim):
        super().__init__()

        self.dense = nn.Dense(int(config.mlp_ratio * dim), dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob) if config.hidden_dropout_prob else nn.Identity()

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SwinLayer(nn.Cell):

    def __init__(self, config, dim, input_resolution, num_heads, shift_size=0):
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.shift_size = shift_size
        self.window_size = config.window_size
        self.input_resolution = input_resolution
        self.layernorm_before = nn.LayerNorm([dim], epsilon=config.layer_norm_eps)
        self.attention = SwinAttention(config, dim, num_heads, window_size=self.window_size)
        self.drop_path = SwinDropPath(config.drop_path_rate) if config.drop_path_rate > 0.0 else nn.Identity()
        self.layernorm_after = nn.LayerNorm([dim], epsilon=config.layer_norm_eps)
        self.intermediate = SwinIntermediate(config, dim)
        self.output = SwinOutput(config, dim)
        # mindcv impl.
        self.roll_pos = Roll(self.shift_size)
        self.roll_neg = Roll(-self.shift_size)
        self.window_partition = WindowPartition(self.window_size)
        self.window_reverse = WindowReverse()

    def set_shift_and_window_size(self, input_resolution):
        if min(input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(input_resolution)

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            img_mask = np.zeros((1, height, width, 1), dtype=np.float32)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = Tensor.from_numpy(mask_windows).astype(dtype)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def maybe_pad(self, hidden_states, height, width):
        pad_right = (self.window_size - width % self.window_size) % self.window_size
        pad_bottom = (self.window_size - height % self.window_size) % self.window_size
        pad_values = (0, 0, 0, pad_right, 0, pad_bottom)
        hidden_states = F.pad(hidden_states, pad_values)
        return hidden_states, pad_values

    def construct(
        self,
        hidden_states: Tensor,
        input_dimensions: Tuple[int, int],
        always_partition: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        if not always_partition:
            self.set_shift_and_window_size(input_dimensions)
        height, width = input_dimensions
        batch_size, _, channels = hidden_states.shape
        shortcut = hidden_states

        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = hidden_states.view(batch_size, height, width, channels)

        # pad hidden_states to multiples of window size
        hidden_states, pad_values = self.maybe_pad(hidden_states, height, width)

        _, height_pad, width_pad, _ = hidden_states.shape
        # cyclic shift
        if self.shift_size > 0:
            #shifted_hidden_states = F.roll(hidden_states, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_hidden_states = self.roll_neg(hidden_states)
        else:
            shifted_hidden_states = hidden_states

        # partition windows
        hidden_states_windows = self.window_partition(shifted_hidden_states)
        hidden_states_windows = hidden_states_windows.view(-1, self.window_size * self.window_size, channels)
        attn_mask = self.get_attn_mask(height_pad, width_pad, dtype=hidden_states.dtype)
        #if attn_mask is not None: attn_mask = attn_mask.to(hidden_states_windows)

        attention_output = self.attention(hidden_states_windows, attn_mask)
        attention_windows = attention_output.view(-1, self.window_size, self.window_size, channels)
        shifted_windows = self.window_reverse(attention_windows, self.window_size, height_pad, width_pad)

        # reverse cyclic shift
        if self.shift_size > 0:
            #attention_windows = F.roll(shifted_windows, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            attention_windows = self.roll_pos(shifted_windows)
        else:
            attention_windows = shifted_windows

        was_padded = pad_values[3] > 0 or pad_values[5] > 0
        if was_padded:
            attention_windows = attention_windows[:, :height, :width, :]    # .contiguous()

        attention_windows = attention_windows.view(batch_size, height * width, channels)
        hidden_states = shortcut + self.drop_path(attention_windows)
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)
        layer_output = hidden_states + self.output(layer_output)
        return layer_output


class SwinStage(nn.Cell):

    def __init__(self, config, dim, input_resolution, depth, num_heads, drop_path, downsample):
        super().__init__()

        self.config = config
        self.dim = dim
        self.blocks = nn.CellList()
        for i in range(depth):
            block = SwinLayer(config, dim, input_resolution, num_heads, (0 if (i % 2 == 0) else config.window_size // 2))
            self.blocks.append(block)
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=nn.LayerNorm) if downsample is not None else None

    def construct(
        self,
        hidden_states: Tensor,
        input_dimensions: Tuple[int, int],
        always_partition: bool = False,
    ) -> Tuple[Tensor]:
        height, width = input_dimensions
        for i, layer_module in enumerate(self.blocks):
            hidden_states = layer_module(hidden_states, input_dimensions, always_partition)

        hidden_states_before_downsampling = hidden_states
        if self.downsample is not None:
            height_downsampled, width_downsampled = (height + 1) // 2, (width + 1) // 2
            output_dimensions = (height, width, height_downsampled, width_downsampled)
            hidden_states = self.downsample(hidden_states_before_downsampling, input_dimensions)
        else:
            output_dimensions = (height, width, height, width)
        return (hidden_states, output_dimensions)


class SwinEncoder(nn.Cell):

    def __init__(self, config, grid_size):
        super().__init__()

        self.num_layers = len(config.depths)
        self.config = config
        dpr = [x.item() for x in F.linspace(0, config.drop_path_rate, sum(config.depths))]
        self.layers = nn.CellList([
            SwinStage(
                config=config,
                dim=int(config.embed_dim * 2**i_layer),
                input_resolution=(grid_size[0] // (2**i_layer), grid_size[1] // (2**i_layer)),
                depth=config.depths[i_layer],
                num_heads=config.num_heads[i_layer],
                drop_path=dpr[sum(config.depths[:i_layer]) : sum(config.depths[: i_layer + 1])],
                downsample=SwinPatchMerging if (i_layer < self.num_layers - 1) else None,
            )
            for i_layer in range(self.num_layers)
        ])

    def construct(
        self,
        hidden_states: Tensor,
        input_dimensions: Tuple[int, int],
        always_partition: bool = False,
    ) -> Tensor:
        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module(hidden_states, input_dimensions, always_partition)
            hidden_states = layer_outputs[0]
            output_dimensions = layer_outputs[1]
            input_dimensions = (output_dimensions[-2], output_dimensions[-1])
        return hidden_states


class SwinModel(nn.Cell):

    def __init__(self, config, add_pooling_layer=True, use_mask_token=False):
        super().__init__()

        self.config = config
        self.num_layers = len(config.depths)
        self.num_features = int(config.embed_dim * 2 ** (self.num_layers - 1))
        self.embeddings = SwinEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = SwinEncoder(config, self.embeddings.patch_grid)
        self.layernorm = nn.LayerNorm([self.num_features], epsilon=config.layer_norm_eps)
        self.pooler = nn.AdaptiveAvgPool1d(1) if add_pooling_layer else None

    def construct(
        self,
        pixel_values: Tensor,
        bool_masked_pos: Tensor = None,
    ) -> Tensor:
        embedding_output, input_dimensions = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        sequence_output = self.encoder(embedding_output, input_dimensions)
        sequence_output = self.layernorm(sequence_output)
        pooled_output = sequence_output
        if self.pooler is not None:
            pooled_output = self.pooler(sequence_output.permute(0, 2, 1))
            pooled_output = F.flatten(pooled_output, start_dim=1)
        return pooled_output


class SwinForImageClassification(nn.Cell):

    def __init__(self, config):
        super().__init__()

        self.num_labels = 2
        self.swin = SwinModel(config)
        self.classifier = nn.Dense(self.swin.num_features, self.num_labels)

    def construct(self, pixel_values: Tensor = None) -> Tensor:
        pooled_output = self.swin(pixel_values)
        logits = self.classifier(pooled_output)
        return logits


def param_dict_name_mapping(kv:Dict[str, Parameter]) -> Dict[str, Parameter]:
    new_kv = {}
    for k in kv.keys():
        new_k = k
        # LayerNorm
        if '.layernorm_before.weight' in k: new_k = k.replace('.layernorm_before.weight', '.layernorm_before.gamma')
        if '.layernorm_before.bias'   in k: new_k = k.replace('.layernorm_before.bias',   '.layernorm_before.beta')
        if '.layernorm_after.weight'  in k: new_k = k.replace('.layernorm_after.weight',  '.layernorm_after.gamma')
        if '.layernorm_after.bias'    in k: new_k = k.replace('.layernorm_after.bias',    '.layernorm_after.beta')
        if '.layernorm.weight'        in k: new_k = k.replace('.layernorm.weight',        '.layernorm.gamma')
        if '.layernorm.bias'          in k: new_k = k.replace('.layernorm.bias',          '.layernorm.beta')
        if '.norm.weight'             in k: new_k = k.replace('.norm.weight',             '.norm.gamma')
        if '.norm.bias'               in k: new_k = k.replace('.norm.bias',               '.norm.beta')
        new_kv[new_k] = kv[k]
    return new_kv


def infer_swin(model:SwinForImageClassification, img:PILImage) -> int:
    X = transform(img)[0]   # do not know why this returns a ndarray tuple
    X = Tensor.from_numpy(X).astype(ms.float32)
    X = X.unsqueeze(0)
    logits = model(X)
    preds = logits.argmax(axis=-1)
    return 1 - preds[0].item()      # swap 0-1


def get_app(app_name:str) -> SwinForImageClassification:
    HF_PATH = Path(__file__).parent.parent / 'huggingface'
    APP_PATH = HF_PATH / app_name
    CONFIG_FILE = APP_PATH / 'model.json'
    WEIGHT_FILE = APP_PATH / 'model.npz'

    with open(CONFIG_FILE) as fh:
        cfg = json.load(fh)
    config = SwinConfig(**cfg)

    model = SwinForImageClassification(config)
    model = model.set_train(False)
    param_dict = load_npz_as_param_dict(WEIGHT_FILE)
    param_dict = param_dict_name_mapping(param_dict)
    ms.load_param_into_net(model, param_dict)
    return model


def get_AI_image_detector() -> SwinForImageClassification:
    return get_app('umm-maybe#AI-image-detector')


def get_xl_detector() -> SwinForImageClassification:
    return get_app('Organikasd#xl-detector')


if __name__ == '__main__':
    model = get_AI_image_detector()
    #model = get_xl_detector()
    X = F.zeros([1, 3, 224, 224])
    logits = model(X)
    print(logits)
