#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/01/03 

from models.utils import *
try: from mindcv.models.resnet import resnet18
except: pass

transform = T.Compose([
    VT.ToTensor(),
    VT.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], is_hwc=False),
])


class ResNetConfig:

    model_type = "resnet"
    layer_types = ["basic", "bottleneck"]

    def __init__(
        self,
        num_channels=3,
        embedding_size=64,
        hidden_sizes=[256, 512, 1024, 2048],
        depths=[3, 4, 6, 3],
        layer_type="bottleneck",
        hidden_act="relu",
        downsample_in_first_stage=False,
        **kwargs,
    ):
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.layer_type = layer_type
        self.hidden_act = hidden_act
        self.downsample_in_first_stage = downsample_in_first_stage


class ResNetConvLayer(nn.Cell):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: str = "relu"):
        super().__init__()

        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, pad_mode='pad', padding=kernel_size // 2, has_bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)
        self.activation = ACT2FN.get(activation, ACT2FN['linear'])

    def construct(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetEmbeddings(nn.Cell):

    def __init__(self, config: ResNetConfig):
        super().__init__()

        self.embedder = ResNetConvLayer(config.num_channels, config.embedding_size, kernel_size=7, stride=2, activation=config.hidden_act)
        self.pooler = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='pad', padding=1)
        self.num_channels = config.num_channels

    def construct(self, pixel_values: Tensor) -> Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError("Make sure that the channel dimension of the pixel values match with the one set in the configuration." )
        embedding = self.embedder(pixel_values)
        embedding = self.pooler(embedding)
        return embedding


class ResNetShortCut(nn.Cell):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()

        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, has_bias=False)
        self.normalization = nn.BatchNorm2d(out_channels)

    def construct(self, input: Tensor) -> Tensor:
        hidden_state = self.convolution(input)
        hidden_state = self.normalization(hidden_state)
        return hidden_state


class ResNetBasicLayer(nn.Cell):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu"):
        super().__init__()

        should_apply_shortcut = in_channels != out_channels or stride != 1
        self.shortcut = ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        self.layer = nn.SequentialCell(
            ResNetConvLayer(in_channels, out_channels, stride=stride),
            ResNetConvLayer(out_channels, out_channels, activation=None),
        )
        self.activation = ACT2FN[activation]

    def construct(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetBottleNeckLayer(nn.Cell):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: str = "relu", reduction: int = 4):
        super().__init__()

        should_apply_shortcut = in_channels != out_channels or stride != 1
        reduces_channels = out_channels // reduction
        self.shortcut = ResNetShortCut(in_channels, out_channels, stride=stride) if should_apply_shortcut else nn.Identity()
        self.layer = nn.SequentialCell(
            ResNetConvLayer(in_channels, reduces_channels, kernel_size=1),
            ResNetConvLayer(reduces_channels, reduces_channels, stride=stride),
            ResNetConvLayer(reduces_channels, out_channels, kernel_size=1, activation=None),
        )
        self.activation = ACT2FN[activation]

    def construct(self, hidden_state):
        residual = hidden_state
        hidden_state = self.layer(hidden_state)
        residual = self.shortcut(residual)
        hidden_state += residual
        hidden_state = self.activation(hidden_state)
        return hidden_state


class ResNetStage(nn.Cell):

    def __init__(self, config: ResNetConfig, in_channels: int, out_channels: int, stride: int = 2, depth: int = 2):
        super().__init__()

        layer = ResNetBottleNeckLayer if config.layer_type == "bottleneck" else ResNetBasicLayer
        self.layers = nn.SequentialCell(
            # downsampling is done in the first layer with stride of 2
            layer(in_channels, out_channels, stride=stride, activation=config.hidden_act),
            *[layer(out_channels, out_channels, activation=config.hidden_act) for _ in range(depth - 1)],
        )

    def construct(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class ResNetEncoder(nn.Cell):

    def __init__(self, config: ResNetConfig):
        super().__init__()

        self.stages = nn.CellList([
            # based on `downsample_in_first_stage` the first layer of the first stage may or may not downsample the input
            ResNetStage(
                config,
                config.embedding_size,
                config.hidden_sizes[0],
                stride=2 if config.downsample_in_first_stage else 1,
                depth=config.depths[0],
            )
        ])
        in_out_channels = zip(config.hidden_sizes, config.hidden_sizes[1:])
        for (in_channels, out_channels), depth in zip(in_out_channels, config.depths[1:]):
            self.stages.append(ResNetStage(config, in_channels, out_channels, depth=depth))

    def construct(self, hidden_state: Tensor) -> Tensor:
        for stage_module in self.stages:
            hidden_state = stage_module(hidden_state)
        return hidden_state


class ResNetModel(nn.Cell):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.embedder = ResNetEmbeddings(config)
        self.encoder = ResNetEncoder(config)
        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

    def construct(self, pixel_values: Tensor):
        embedding_output = self.embedder(pixel_values)
        encoder_outputs = self.encoder(embedding_output)
        pooled_output = self.pooler(encoder_outputs)
        return pooled_output


class ResNetForImageClassification(nn.Cell):

    def __init__(self, config):
        super().__init__()

        self.num_labels = 2
        self.resnet = ResNetModel(config)
        self.classifier = nn.SequentialCell(
            nn.Flatten(),
            nn.Dense(config.hidden_sizes[-1], self.num_labels),
        )

    def construct(self, pixel_values: Tensor) -> Tensor:
        outputs = self.resnet(pixel_values)
        logits = self.classifier(outputs)
        return logits


def param_dict_name_mapping(kv:Dict[str, Parameter]) -> Dict[str, Parameter]:
    new_kv = {}
    for k in kv.keys():
        new_k = k
        # BatchNorm
        if k.endswith('.normalization.running_mean'): new_k = k.replace('.normalization.running_mean', '.normalization.moving_mean')
        if k.endswith('.normalization.running_var'):  new_k = k.replace('.normalization.running_var',  '.normalization.moving_variance')
        if k.endswith('.normalization.weight'):       new_k = k.replace('.normalization.weight',       '.normalization.gamma')
        if k.endswith('.normalization.bias'):         new_k = k.replace('.normalization.bias',         '.normalization.beta')
        if k.endswith('.num_batches_tracked'): continue     # ignore key
        new_kv[new_k] = kv[k]
    return new_kv


def infer_resnet(model:nn.Cell, img:PILImage, debug:bool=False) -> Union[int, Tuple[float, float], Tuple[int, int], Tuple[int]]:
    X = transform(img)[0]   # do not know why this returns a ndarray tuple
    X = Tensor.from_numpy(X).astype(ms.float32)
    logits = model(X.unsqueeze(0)).squeeze(0)
    probs = F.softmax(logits, axis=-1)
    pred = F.argmax(probs).item()
    return (logits.numpy().tolist(), probs.numpy().tolist(), pred) if debug else pred


def get_app(app_name:str) -> ResNetForImageClassification:
    HF_PATH = Path(__file__).parent.parent / 'huggingface'
    APP_PATH = HF_PATH / app_name
    CONFIG_FILE = APP_PATH / 'model.json'
    WEIGHT_FILE = APP_PATH / 'model.npz'

    with open(CONFIG_FILE) as fh:
        cfg = json.load(fh)
    config = ResNetConfig(**cfg)

    model = ResNetForImageClassification(config)
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


def get_resnet18_finetuned_ai_art() -> ResNetForImageClassification:
    return get_app('artfan123#resnet-18-finetuned-ai-art')


if __name__ == '__main__':
    model = get_resnet18_finetuned_ai_art()
    print(model)
    X = F.zeros([1, 3, 224, 224])
    logits = model(X)
    print(logits)
