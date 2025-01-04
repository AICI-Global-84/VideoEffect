import os
import io
import tempfile
import hashlib
import json
import numpy as np
import soundfile as sf
import torch

import torchaudio
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeVideoClip,
    concatenate_audioclips,
    AudioClip,
    ImageClip,
    VideoClip
)
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from PIL import Image, ImageDraw

class AudioSoundWavesEffect:
    """
    Node: AudioSoundWavesEffect
    Tạo hiệu ứng sound waves (cột/ sóng) đè lên video.
    Upload video đã xử lý lên Google Drive, trả về link public và payload video byte.
    """

    SCOPES = ['https://www.googleapis.com/auth/drive.file']
    SERVICE_ACCOUNT_FILE = '/content/drive/My Drive/SD-Data/comfyui-n8n-aici01-7679b55c962b.json'
    DRIVE_FOLDER_ID = '1fZyeDT_eW6ozYXhqi_qLVy-Xnu5JD67a'
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
                "wave_type": (["columns", "wave"],),
                "wave_position": (["center", "top_third", "bottom_third"],),
                "wave_width": ("INT", {"default": 3, "min": 1, "max": 50, "step": 1}),
                "wave_height": ("INT", {"default": 100, "min": 10, "max": 1000, "step": 10}),
                "wave_color": ("STRING", {"default": "#FF0000"}),  # mã màu hex
                "start_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                }),
                "end_duration": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "display": "number"
                })
            }
        }

    RETURN_TYPES = ("STRING", "VIDEO",)
    RETURN_NAMES = ("file_url", "combined_video",)
    FUNCTION = "combine_audio_soundwave"
    CATEGORY = "video/audio"

    def __init__(self):
        self.drive_service = None
        self._initialize_drive_service()

    def _initialize_drive_service(self):
        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.SERVICE_ACCOUNT_FILE, scopes=self.SCOPES
            )
            self.drive_service = build('drive', 'v3', credentials=credentials)
        except Exception as e:
            print(f"Error initializing Drive service: {str(e)}")
            raise RuntimeError(f"Failed to initialize Drive service: {str(e)}")

    def _upload_to_drive(self, file_path):
        """
        Upload file_path lên Google Drive, thiết lập quyền 'reader' cho bất kỳ ai có link.
        Trả về link xem công khai.
        """
        try:
            file_metadata = {
                'name': os.path.basename(file_path),
                'parents': [self.DRIVE_FOLDER_ID]
            }
            media = MediaFileUpload(file_path, resumable=True)
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()

            # Cấp quyền đọc cho anyone
            self.drive_service.permissions().create(
                fileId=file.get('id'),
                body={'type': 'anyone', 'role': 'reader'},
                fields='id'
            ).execute()

            file_id = file.get('id')
            return f"https://drive.google.com/uc?id={file_id}"

        except Exception as e:
            raise RuntimeError(f"Failed to upload to Drive: {str(e)}")

    def combine_audio_soundwave(
        self,
        video,
        audio,
        wave_type,
        wave_position,
        wave_width,
        wave_height,
        wave_color,
        start_duration,
        end_duration
    ):
        """
        1) Xử lý video, audio (chèn silent audio ở đầu, cắt video cho khớp).
        2) Vẽ hiệu ứng waves (theo wave_type) overlay lên video.
        3) Upload lên Google Drive, trả về link + payload.
        """
        temp_output_path = None

        try:
            # --- Xử lý video input ---
            if isinstance(video, bytes):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
                    temp_video_file.write(video)
                    temp_video_path = temp_video_file.name
                    video_clip = VideoFileClip(temp_video_path)
            else:
                raise TypeError(f"Invalid video input: {type(video)}")

            # --- Xử lý audio input ---
            if isinstance(audio, dict) and 'waveform' in audio and 'sample_rate' in audio:
                waveform = audio['waveform'].numpy().squeeze()  # tensor -> numpy & remove extra dim
                sample_rate = audio['sample_rate']
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
                    sf.write(temp_audio_file.name, waveform, sample_rate)
                    temp_audio_path = temp_audio_file.name

                audio_clip = AudioFileClip(temp_audio_path)
            else:
                raise TypeError(f"Invalid audio input: {type(audio)}")

            # --- Tổng thời lượng ---
            total_duration = start_duration + audio_clip.duration + end_duration

            # Cắt hoặc lặp video để có duration = total_duration
            if video_clip.duration < total_duration:
                video_clip = video_clip.loop(duration=total_duration)
            else:
                video_clip = video_clip.subclip(0, total_duration)

            # Tạo silent audio ở đầu
            silent_audio = AudioClip(lambda t: 0, duration=start_duration, fps=audio_clip.fps)
            final_audio_clip = concatenate_audioclips([silent_audio, audio_clip])

            # Kết hợp audio vào video
            base_clip = CompositeVideoClip([video_clip]).set_audio(final_audio_clip)

            # --- Tạo clip waves (overlay) ---
            # Để dễ thêm kiểu effect khác sau này, ta tách hàm tạo wave_clip
            wave_clip = self._create_wave_overlay(
                base_clip.size,            # (width, height) của video
                wave_type,
                waveform,
                sample_rate,
                wave_color,
                wave_width,
                wave_height,
                wave_position,
                total_duration
            )

            # Overlay wave_clip lên video
            final_clip = CompositeVideoClip([base_clip, wave_clip])

            # --- Xuất video ---
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
                temp_output_path = temp_output.name
            final_clip.write_videofile(
                temp_output_path,
                codec='libx264',
                audio_codec='aac',
                remove_temp=True
            )
            final_clip.close()
            video_clip.close()
            audio_clip.close()

            # --- Upload lên Drive ---
            drive_url = self._upload_to_drive(temp_output_path)

            # --- Chuẩn bị payload byte ---
            with open(temp_output_path, 'rb') as f:
                combined_video = {'path': temp_output_path, 'data': f.read()}

            return (drive_url, combined_video)

        except Exception as e:
            raise RuntimeError(f"Failed to create and upload waves video: {str(e)}")

        finally:
            if temp_output_path and os.path.exists(temp_output_path):
                os.unlink(temp_output_path)

    def _create_wave_overlay(
        self,
        video_size,
        wave_type,
        waveform,
        sample_rate,
        wave_color,
        wave_width,
        wave_height,
        wave_position,
        duration
    ):
        """
        Tạo một VideoClip 'waves' bằng cách vẽ frame-by-frame theo thời gian.
        wave_type = 'columns' hoặc 'wave' (đơn giản).
        """
        w, h = video_size
        fps = 24  # Tùy chọn, bạn có thể điều chỉnh fps nếu muốn

        # Chuyển mã màu hex -> (R, G, B)
        wave_color_rgb = self._hex_to_rgb(wave_color)

        # Tính vị trí y cho waves dựa vào wave_position
        if wave_position == "center":
            wave_y_mid = h // 2
        elif wave_position == "top_third":
            wave_y_mid = h // 3
        elif wave_position == "bottom_third":
            wave_y_mid = (h * 2) // 3
        else:
            wave_y_mid = h // 2  # mặc định center

        # Chuẩn bị data audio, cắt/zero-pad để trùng với duration (nếu cần)
        # Số mẫu = sample_rate * duration (xấp xỉ)
        total_samples = int(sample_rate * duration)
        if len(waveform) < total_samples:
            # zero-pad
            pad_len = total_samples - len(waveform)
            waveform = np.concatenate([waveform, np.zeros(pad_len)])
        elif len(waveform) > total_samples:
            waveform = waveform[:total_samples]

        # Hàm tạo frame: moviepy sẽ gọi make_frame(t)
        def make_frame(t):
            # Tính index sample trong waveform (để lấy amplitude)
            # Mỗi giây t -> index ~ t * sample_rate
            index = int(t * sample_rate)
            if index < 0 or index >= len(waveform):
                amp = 0
            else:
                amp = waveform[index]

            # Chuẩn bị một ảnh trắng trong suốt (RGBA) để vẽ wave
            frame_img = Image.new("RGBA", (w, h), (255, 255, 255, 0))
            draw = ImageDraw.Draw(frame_img)

            # Chọn kiểu wave
            if wave_type == "columns":
                # Dạng cột (vertical bar). Amp lớn => cột cao.
                # amp ~ [-1..1], ta scale theo wave_height
                col_height = abs(amp) * wave_height
                x_center = w // 2
                y1 = int(wave_y_mid - col_height/2)
                y2 = int(wave_y_mid + col_height/2)
                draw.rectangle(
                    [(x_center - wave_width//2, y1),
                     (x_center + wave_width//2, y2)],
                    fill=wave_color_rgb
                )
            elif wave_type == "wave":
                # Dạng đường sóng ngang (đơn giản).
                # Ta vẽ 1 đường line, offset theo amp
                amplitude = int(amp * wave_height)
                x1 = 0
                x2 = w
                y = wave_y_mid + amplitude
                draw.line([(x1, wave_y_mid), (x2, y)],
                          fill=wave_color_rgb, width=wave_width)

            # Convert PIL -> np array RGBA
            return np.array(frame_img)

        wave_clip = VideoClip(make_frame, duration=duration).set_duration(duration).set_fps(fps)
        return wave_clip

    def _hex_to_rgb(self, hex_color):
        """
        Chuyển mã hex (#RRGGBB) thành (R,G,B).
        Nếu không parse được, trả về màu đỏ mặc định.
        """
        try:
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return (r, g, b, 255)
        except:
            pass
        return (255, 0, 0, 255)  # default red if fail


NODE_CLASS_MAPPINGS = {
    "AudioSoundWavesEffect": AudioSoundWavesEffect
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSoundWavesEffect": "Audio Sound Waves Effect"
}
