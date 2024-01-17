#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/10/26

from argparse import ArgumentParser
from numpy import fft

from models.utils import *
from utils import *


def stats_hw():
  fp = IMG_PATH / 'img_size.png'
  if fp.exists(): return

  ws, hs = [], []
  fps = list(DATA_PATH.iterdir())
  for fp in tqdm(fps):
    img = load_img(fp)
    w, h = img.size
    ws.append(w)
    hs.append(h)
  plt.scatter(ws, hs, s=1)
  plt.xlabel('width')
  plt.ylabel('hieght')
  plt.savefig(fp, dpi=600)

  print('min(width):',  min(ws))
  print('min(height):', min(hs))
  print('max(width):',  max(ws))
  print('max(height):', max(hs))


def plot_hi_lo(img:PILImage):
  img_lo = img.filter(ImageFilter.GaussianBlur(3))
  im_lo = pil_to_npimg(img_lo)
  im = pil_to_npimg(img)
  im_hi = minmax_norm(npimg_diff(im, im_lo))

  plt.clf()
  plt.subplot(1, 3, 1) ; plt.axis('off') ; plt.title('X')    ; plt.imshow(im)
  plt.subplot(1, 3, 2) ; plt.axis('off') ; plt.title('X_lo') ; plt.imshow(im_lo)
  plt.subplot(1, 3, 3) ; plt.axis('off') ; plt.title('X_hi') ; plt.imshow(im_hi)
  plt.show()


def plot_fft(img:PILImage):
  img_grey = img.convert('L')
  amp = np.log(np.abs(fft.fftshift(fft.fft2(img_grey))))
  im = pil_to_npimg(img).astype(np.float32) / 255.0
  freq = np.stack([fft.fftshift(fft.fft2(im[:, :, i])) for i in range(im.shape[-1])], axis=-1)

  r = 30
  H, W, C = freq.shape
  Hc, Wc = H // 2, W // 2
  freq_lo = np.zeros_like(freq)
  freq_lo[Hc-r:Hc+r, Wc-r:Wc+r, :] = freq[Hc-r:Hc+r, Wc-r:Wc+r, :]
  freq_hi = freq - freq_lo
  im_lo = np.stack([fft.ifft2(fft.ifftshift(freq_lo[:, :, i])) for i in range(C)], axis=-1)
  im_hi = np.stack([fft.ifft2(fft.ifftshift(freq_hi[:, :, i])) for i in range(C)], axis=-1)
  im_lo = np.abs(im_lo)
  im_hi = np.abs(im_hi)

  plt.clf()
  plt.subplot(2, 2, 1) ; plt.imshow(im)    ; plt.title('X')
  plt.subplot(2, 2, 2) ; sns.heatmap(amp, cbar=True) ; plt.title('freq')
  plt.subplot(2, 2, 3) ; plt.imshow(im_lo) ; plt.title('low')
  plt.subplot(2, 2, 4) ; plt.imshow(im_hi) ; plt.title('high')
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--hw',  action='store_true')
  parser.add_argument('--hf',  action='store_true')
  parser.add_argument('--fft', action='store_true')
  args = parser.parse_args()

  if args.hw:
    stats_hw()

  if args.hf:
    try:
      for fp in DATA_PATH.iterdir():
        img = load_img(fp)
        plot_hi_lo(img)
    except KeyboardInterrupt:
      pass

  if args.fft:
    try:
      for fp in DATA_PATH.iterdir():
        img = load_img(fp)
        plot_fft(img)
    except KeyboardInterrupt:
      pass
