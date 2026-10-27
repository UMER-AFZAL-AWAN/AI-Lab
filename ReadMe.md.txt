python inference_video.py --video path_to_your_video.mp4 --out path_to_output.mp4


python inference_video.py --C:\AI-Lab\InputVideo\input.mp4 --out C:\AI-Lab\ECCV2022-RIFE-main\OUTPUT\output.mp4


input 26 sec outpu 12 
ffmpeg -hwaccel cuda -i input.mp4 -vf "mpdecimate,setpts=N/FRAME_RATE/TB,minterpolate=fps=60:mi_mode=mci" -c:v h264_nvenc -preset slow -an output_smooth_fixed.mp4 





ffmpeg -hwaccel cuda -i input.mp4 -vf "minterpolate=fps=60:mi_mode=mci" -c:v h264_nvenc -preset slow -an output_smooth_26s_direct.mp4




Microsoft Windows [Version 10.0.26200.9711]
(c) Microsoft Corporation. All rights reserved.

C:\AI-Lab\InputVideo>ffmpeg -i input.mp4
ffmpeg version 2026-05-28-git-7b46c6a2a3-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers
  built with gcc 16.1.0 (Rev2, Built by MSYS2 project)
  configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autodetect --enable-cairo --enable-fontconfig --enable-iconv --enable-gnutls --enable-lcms2 --enable-libxml2 --enable-gmp --enable-bzlib --enable-lzma --enable-libsnappy --enable-zlib --enable-librist --enable-libsrt --enable-libssh --enable-libzmq --enable-avisynth --enable-libbluray --enable-libcaca --enable-libdvdnav --enable-libdvdread --enable-sdl2 --enable-libaribb24 --enable-libaribcaption --enable-libdav1d --enable-libdavs2 --enable-libopenjpeg --enable-libquirc --enable-libuavs3d --enable-libxevd --enable-libzvbi --enable-liboapv --enable-libqrencode --enable-librav1e --enable-libsvtav1 --enable-libvvenc --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxavs2 --enable-libxeve --enable-libxvid --enable-libaom --enable-libjxl --enable-libsvtjpegxs --enable-libvpx --enable-mediafoundation --enable-libass --enable-frei0r --enable-libfreetype --enable-libfribidi --enable-libharfbuzz --enable-liblensfun --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda-llvm --enable-cuvid --enable-dxva2 --enable-d3d11va --enable-d3d12va --enable-ffnvcodec --enable-libvpl --enable-nvdec --enable-nvenc --enable-vaapi --enable-libshaderc --enable-vulkan --enable-libplacebo --enable-opencl --enable-libcdio --enable-openal --enable-libgme --enable-libmodplug --enable-libopenmpt --enable-libopencore-amrwb --enable-libmp3lame --enable-libshine --enable-libtheora --enable-libtwolame --enable-libvo-amrwbenc --enable-libcodec2 --enable-libilbc --enable-libgsm --enable-liblc3 --enable-libopencore-amrnb --enable-libopus --enable-libspeex --enable-libvorbis --enable-ladspa --enable-libbs2b --enable-libflite --enable-libmysofa --enable-librubberband --enable-libsoxr --enable-chromaprint --enable-whisper
  libavutil      60. 31.100 / 60. 31.100
  libavcodec     62. 34.102 / 62. 34.102
  libavformat    62. 18.100 / 62. 18.100
  libavdevice    62.  4.100 / 62.  4.100
  libavfilter    11. 17.100 / 11. 17.100
  libswscale      9.  7.100 /  9.  7.100
  libswresample   6.  4.100 /  6.  4.100
[in#0 @ 000001c592c20700] Skipping unhandled metadata com.video.file.type of type 67
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'input.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2026-01-07T11:08:18.000000Z
    com.android.version: 15
  Duration: 00:00:26.56, start: 0.000000, bitrate: 4946 kb/s
  Stream #0:0[0x1](eng): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720, 4942 kb/s, 30.27 fps, 59.94 tbr, 90k tbn (default)
    Metadata:
      creation_time   : 2026-01-07T11:08:18.000000Z
      handler_name    : VideoHandle
At least one output file must be specified

C:\AI-Lab\InputVideo>ffmpeg -hwaccel cuda -i input.mp4 -vf "mpdecimate,setpts=N/FRAME_RATE/TB,minterpolate=fps=60:mi_mode=mci" -c:v h264_nvenc -preset slow -an output_smooth_fixed.mp4
ffmpeg version 2026-05-28-git-7b46c6a2a3-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers
  built with gcc 16.1.0 (Rev2, Built by MSYS2 project)
  configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autodetect --enable-cairo --enable-fontconfig --enable-iconv --enable-gnutls --enable-lcms2 --enable-libxml2 --enable-gmp --enable-bzlib --enable-lzma --enable-libsnappy --enable-zlib --enable-librist --enable-libsrt --enable-libssh --enable-libzmq --enable-avisynth --enable-libbluray --enable-libcaca --enable-libdvdnav --enable-libdvdread --enable-sdl2 --enable-libaribb24 --enable-libaribcaption --enable-libdav1d --enable-libdavs2 --enable-libopenjpeg --enable-libquirc --enable-libuavs3d --enable-libxevd --enable-libzvbi --enable-liboapv --enable-libqrencode --enable-librav1e --enable-libsvtav1 --enable-libvvenc --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxavs2 --enable-libxeve --enable-libxvid --enable-libaom --enable-libjxl --enable-libsvtjpegxs --enable-libvpx --enable-mediafoundation --enable-libass --enable-frei0r --enable-libfreetype --enable-libfribidi --enable-libharfbuzz --enable-liblensfun --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda-llvm --enable-cuvid --enable-dxva2 --enable-d3d11va --enable-d3d12va --enable-ffnvcodec --enable-libvpl --enable-nvdec --enable-nvenc --enable-vaapi --enable-libshaderc --enable-vulkan --enable-libplacebo --enable-opencl --enable-libcdio --enable-openal --enable-libgme --enable-libmodplug --enable-libopenmpt --enable-libopencore-amrwb --enable-libmp3lame --enable-libshine --enable-libtheora --enable-libtwolame --enable-libvo-amrwbenc --enable-libcodec2 --enable-libilbc --enable-libgsm --enable-liblc3 --enable-libopencore-amrnb --enable-libopus --enable-libspeex --enable-libvorbis --enable-ladspa --enable-libbs2b --enable-libflite --enable-libmysofa --enable-librubberband --enable-libsoxr --enable-chromaprint --enable-whisper
  libavutil      60. 31.100 / 60. 31.100
  libavcodec     62. 34.102 / 62. 34.102
  libavformat    62. 18.100 / 62. 18.100
  libavdevice    62.  4.100 / 62.  4.100
  libavfilter    11. 17.100 / 11. 17.100
  libswscale      9.  7.100 /  9.  7.100
  libswresample   6.  4.100 /  6.  4.100
[in#0 @ 00000225a3751500] Skipping unhandled metadata com.video.file.type of type 67
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'input.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2026-01-07T11:08:18.000000Z
    com.android.version: 15
  Duration: 00:00:26.56, start: 0.000000, bitrate: 4946 kb/s
  Stream #0:0[0x1](eng): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720, 4942 kb/s, 30.27 fps, 59.94 tbr, 90k tbn (default)
    Metadata:
      creation_time   : 2026-01-07T11:08:18.000000Z
      handler_name    : VideoHandle
Stream mapping:
  Stream #0:0 -> #0:0 (h264 (native) -> h264 (h264_nvenc))
Press [q] to stop, [?] for help
Output #0, mp4, to 'output_smooth_fixed.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    com.android.version: 15
    encoder         : Lavf62.18.100
  Stream #0:0(eng): Video: h264 (Main) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720, q=2-31, 2000 kb/s, 60 fps, 15360 tbn (default)
    Metadata:
      encoder         : Lavc62.34.102 h264_nvenc
      creation_time   : 2026-01-07T11:08:18.000000Z
      handler_name    : VideoHandle
    Side data:
      CPB properties: bitrate max/min/avg: 0/0/2000000 buffer size: 4000000 vbv_delay: N/A
[out#0/mp4 @ 00000225a37daf40] video:3398KiB audio:0KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: 0.287952%
frame=  758 fps=7.7 q=34.0 Lsize=    3408KiB time=00:00:12.58 bitrate=2218.4kbits/s speed=0.129x elapsed=0:01:37.83

C:\AI-Lab\InputVideo>ffmpeg -hwaccel cuda -i input.mp4 -vf "minterpolate=fps=60:mi_mode=mci" -c:v h264_nvenc -preset slow -an output_smooth_26s_direct.mp4
ffmpeg version 2026-05-28-git-7b46c6a2a3-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers
  built with gcc 16.1.0 (Rev2, Built by MSYS2 project)
  configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autodetect --enable-cairo --enable-fontconfig --enable-iconv --enable-gnutls --enable-lcms2 --enable-libxml2 --enable-gmp --enable-bzlib --enable-lzma --enable-libsnappy --enable-zlib --enable-librist --enable-libsrt --enable-libssh --enable-libzmq --enable-avisynth --enable-libbluray --enable-libcaca --enable-libdvdnav --enable-libdvdread --enable-sdl2 --enable-libaribb24 --enable-libaribcaption --enable-libdav1d --enable-libdavs2 --enable-libopenjpeg --enable-libquirc --enable-libuavs3d --enable-libxevd --enable-libzvbi --enable-liboapv --enable-libqrencode --enable-librav1e --enable-libsvtav1 --enable-libvvenc --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxavs2 --enable-libxeve --enable-libxvid --enable-libaom --enable-libjxl --enable-libsvtjpegxs --enable-libvpx --enable-mediafoundation --enable-libass --enable-frei0r --enable-libfreetype --enable-libfribidi --enable-libharfbuzz --enable-liblensfun --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda-llvm --enable-cuvid --enable-dxva2 --enable-d3d11va --enable-d3d12va --enable-ffnvcodec --enable-libvpl --enable-nvdec --enable-nvenc --enable-vaapi --enable-libshaderc --enable-vulkan --enable-libplacebo --enable-opencl --enable-libcdio --enable-openal --enable-libgme --enable-libmodplug --enable-libopenmpt --enable-libopencore-amrwb --enable-libmp3lame --enable-libshine --enable-libtheora --enable-libtwolame --enable-libvo-amrwbenc --enable-libcodec2 --enable-libilbc --enable-libgsm --enable-liblc3 --enable-libopencore-amrnb --enable-libopus --enable-libspeex --enable-libvorbis --enable-ladspa --enable-libbs2b --enable-libflite --enable-libmysofa --enable-librubberband --enable-libsoxr --enable-chromaprint --enable-whisper
  libavutil      60. 31.100 / 60. 31.100
  libavcodec     62. 34.102 / 62. 34.102
  libavformat    62. 18.100 / 62. 18.100
  libavdevice    62.  4.100 / 62.  4.100
  libavfilter    11. 17.100 / 11. 17.100
  libswscale      9.  7.100 /  9.  7.100
  libswresample   6.  4.100 /  6.  4.100
[in#0 @ 000001ec4c673e40] Skipping unhandled metadata com.video.file.type of type 67
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'input.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    creation_time   : 2026-01-07T11:08:18.000000Z
    com.android.version: 15
  Duration: 00:00:26.56, start: 0.000000, bitrate: 4946 kb/s
  Stream #0:0[0x1](eng): Video: h264 (High) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720, 4942 kb/s, 30.27 fps, 59.94 tbr, 90k tbn (default)
    Metadata:
      creation_time   : 2026-01-07T11:08:18.000000Z
      handler_name    : VideoHandle
Stream mapping:
  Stream #0:0 -> #0:0 (h264 (native) -> h264 (h264_nvenc))
Press [q] to stop, [?] for help
Output #0, mp4, to 'output_smooth_26s_direct.mp4':
  Metadata:
    major_brand     : mp42
    minor_version   : 0
    compatible_brands: isommp42
    com.android.version: 15
    encoder         : Lavf62.18.100
  Stream #0:0(eng): Video: h264 (Main) (avc1 / 0x31637661), yuv420p(tv, bt709, progressive), 1280x720, q=2-31, 2000 kb/s, 60 fps, 15360 tbn (default)
    Metadata:
      encoder         : Lavc62.34.102 h264_nvenc
      creation_time   : 2026-01-07T11:08:18.000000Z
      handler_name    : VideoHandle
    Side data:
      CPB properties: bitrate max/min/avg: 0/0/2000000 buffer size: 4000000 vbv_delay: N/A
[out#0/mp4 @ 000001ec4c6f5bc0] video:6955KiB audio:0KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: 0.281326%
frame= 1590 fps=9.3 q=33.0 Lsize=    6974KiB time=00:00:26.45 bitrate=2160.0kbits/s speed=0.155x elapsed=0:02:50.23

C:\AI-Lab\InputVideo>