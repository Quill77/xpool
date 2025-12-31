"""
Used to compress video and change frame rate in: https://github.com/ArrowLuo/CLIP4Clip 
Author: ArrowLuo
"""
import os
import argparse
import subprocess
import time
import multiprocessing
from multiprocessing import Pool
import shutil
try:
    from psutil import cpu_count
except:
    from multiprocessing import cpu_count
# multiprocessing.freeze_support()

def compress(paras):
    input_video_path, output_video_path = paras
    try:
        # Check if the input video has an audio stream
        probe_command = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_video_path
        ]
        probe = subprocess.Popen(probe_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        probe_out, probe_err = probe.communicate()
        has_audio = b'audio' in probe_out

        command = [
            'ffmpeg',
            '-y',  # (optional) overwrite output file if it exists
            '-i', input_video_path,
            '-filter:v',
            'setpts=2.5*PTS',  # slow down video to 2fps (2.5 times slower)
            '-r', '2',  # set the output frame rate to 2fps
        ]
        if has_audio:
            command.extend([
                '-filter:a',
                'atempo=0.4',  # slow down audio to match video (0.4 times slower)
                '-map', '0:v',
                '-map', '0:a',
                '-c:a', 'aac',  # 音频编码器
                '-b:a', '128k',  # 音频比特率
            ])
        else:
            command.extend([
                '-map', '0:v',
            ])
        command.extend([
            '-c:v', 'libx264',  # 视频编码器
            '-crf', '18',  # 压缩质量
            output_video_path,
        ])
        ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = ffmpeg.communicate()
        retcode = ffmpeg.poll()
        if retcode != 0:
            print(f"Error processing {input_video_path}: {err.decode('utf-8')}")
        else:
            print(f"Processed {input_video_path}")
    except Exception as e:
        raise e

def prepare_input_output_pairs(input_root, output_root):
    input_video_path_list = []
    output_video_path_list = []
    for root, dirs, files in os.walk(input_root):
        for file_name in files:
            input_video_path = os.path.join(root, file_name)
            output_video_path = os.path.join(output_root, file_name)
            if os.path.exists(output_video_path):
                pass
            else:
                input_video_path_list.append(input_video_path)
                output_video_path_list.append(output_video_path)
    return input_video_path_list, output_video_path_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compress video and change frame rate for speed-up')
    parser.add_argument('--input_root', type=str, help='input root')
    parser.add_argument('--output_root', type=str, help='output root')
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root

    assert input_root != output_root

    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)

    input_video_path_list, output_video_path_list = prepare_input_output_pairs(input_root, output_root)

    print("Total video need to process: {}".format(len(input_video_path_list)))
    num_works = cpu_count()
    print("Begin with {}-core logical processor.".format(num_works))

    pool = Pool(num_works)
    pool.map(compress,
             [(input_video_path, output_video_path) for
              input_video_path, output_video_path in
              zip(input_video_path_list, output_video_path_list)])
    pool.close()
    pool.join()