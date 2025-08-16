14.08.25
Got suggested to run pix2pix on GPU. I had access to a RX5600m. Started on WSL
Realized GPU passthrough does not work WSL, switched to ubuntu

15.08.25
Booted ubuntu, laptop does not connect to university WiFi on linux.
Setup under hotspot. Fiddled with ROCm, downloaded, redownloaded, compiled drivers?
Cryptic errors just won't stop. System crash, GPU undetected. 28GB downloads for ROCm drivers. thrice. under hotspot that barely breaks 5MBps.
Decided to use docker. ROCm+Pytorch image was 22GB. Switched back to windows, downloaded, archiver, back to ubuntu.
Installed docker, unpacked image, loading image to docker. Ran out of storage. turns out 80gb isn't enough.
Fiddled with docker for a while before giving up and returning to bare metal.
Cleaned up drivers, restarted.
Watched a tutorial to retry ROCm+Pytorch (https://phazertech.com/tutorials/rocm.html). Switched out 28GB full download for just the ROCm runtime.

16.08.25
ROCm works. Tried running matrix mult. VRAM and GPU usage confirmed
GPU compute available, but compatibility issues persist on RDNA1. Crashes, lockups.
Tried running pix2pix, instructions outdated.
Messed about to find the correct parameters, installed requirements skipped by conda. downloaded samples, and pretrained model.
Model loads, but does not proceed to output.
Crashed and locks up a few times.
Gave up, trid CPU. Model runs, processes image fine.
Tests google colab for training, instead of local training.
Tests for resuming, and reloading checkpoints.


