# CUDA ED25519 Public Key Vanity Search for Meshcore

Thanks for [Allespro/ed25519_cuda](https://github.com/Allespro/ed25519_cuda) creating the original cuda code for this!!

A friend of mine showed me this website <https://gessaman.com/mc-keygen/> to find vanity keys for [Meshcore](https://meshcore.co.uk/) devices. 
I have no idea about Meshcore nor anything else really, so the most reasonable thing when I saw that it searches for only around 4-10k/s keys, is that I had the urge to improve the speed. 

So I started with rust: [Nachtalb/meshcore-vanity-ed25519](https://github.com/Nachtalb/meshcore-vanity-ed25519) which sped things up to 700k/s on the same machine, and 1.5m/s on a desktop. 

That was still too slow. So I asked my GPU to do it. Thanks to the effort of this origin repo mentiond at the top. I was able to implement vanitiy search with CUDA acceleration. 

Now on the same machine (1.5m/s) we are currently sitting at a comfy **34.5m/s keys.**

## Build 

To build the tool you need
- [CUDA 13](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
- [MSVC 2022+](https://visualstudio.microsoft.com/downloads/)

```ps
nvcc -o vanity.exe vanity.cu -arch=sm_89 -O3 -use_fast_math -lcurand -ladvapi32
```

## Usage 

```ps
vanity.exe [PREFIX]
```

## License
I retained the same zlib license as the original author.
