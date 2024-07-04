import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List, Optional

import onnxruntime
from tqdm import tqdm

import faceswapper.globals
from faceswapper import wording

TEMP_DIRECTORY = '.temp'
TEMP_VIDEO_FILE = 'temp.mp4'

# monkey patch ssl
if platform.system().lower() == 'darwin':
	ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args : List[str]) -> bool:
	commands = [ 'ffmpeg', '-hide_banner', '-loglevel', 'error' ]
	commands.extend(args)
	try:
		subprocess.check_output(commands, stderr = subprocess.STDOUT)
		return True
	except subprocess.CalledProcessError:
		return False


def detect_fps(target_path : str) -> float:
	commands = [ 'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers = 1:nokey = 1', target_path ]
	output = subprocess.check_output(commands).decode().strip().split('/')
	try:
		numerator, denominator = map(int, output)
		return numerator / denominator
	except (ValueError, ZeroDivisionError):
		return 30


def extract_frames(target_path : str, fps : float = 30) -> bool:
	temp_directory_path = get_temp_directory_path(target_path)
	temp_frame_quality = round(31 - (faceswapper.globals.temp_frame_quality * 0.31))
	trim_frame_start = faceswapper.globals.trim_frame_start
	trim_frame_end = faceswapper.globals.trim_frame_end
	commands = [ '-hwaccel', 'auto', '-i', target_path, '-q:v', str(temp_frame_quality), '-pix_fmt', 'rgb24' ]
	if trim_frame_start is not None and trim_frame_end is not None:
		commands.extend(['-vf', 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(trim_frame_end) + ',fps=' + str(fps)])
	elif trim_frame_start is not None:
		commands.extend(['-vf', 'trim=start_frame=' + str(trim_frame_start) + ',fps=' + str(fps)])
	elif trim_frame_end is not None:
		commands.extend(['-vf', 'trim=end_frame=' + str(trim_frame_end) + ',fps=' + str(fps)])
	else:
		commands.extend(['-vf', 'fps=' + str(fps)])
	commands.extend([os.path.join(temp_directory_path, '%04d.' + faceswapper.globals.temp_frame_format)])
	return run_ffmpeg(commands)


def create_video(target_path : str, fps : float = 30) -> bool:
	temp_output_path = get_temp_output_path(target_path)
	temp_directory_path = get_temp_directory_path(target_path)
	output_video_quality = round(51 - (faceswapper.globals.output_video_quality * 0.5))
	commands = [ '-hwaccel', 'auto', '-r', str(fps), '-i', os.path.join(temp_directory_path, '%04d.' + faceswapper.globals.temp_frame_format), '-c:v', faceswapper.globals.output_video_encoder ]
	if faceswapper.globals.output_video_encoder in [ 'libx264', 'libx265', 'libvpx' ]:
		commands.extend(['-crf', str(output_video_quality)])
	if faceswapper.globals.output_video_encoder in [ 'h264_nvenc', 'hevc_nvenc' ]:
		commands.extend([ '-cq', str(output_video_quality) ])
	commands.extend([ '-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625', '-y', temp_output_path ])
	return run_ffmpeg(commands)