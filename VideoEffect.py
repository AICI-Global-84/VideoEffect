import os
import cv2
import numpy as np
from pydub import AudioSegment
import math

# =============================================================================
# Node AudioSoundWavesEffect
# =============================================================================
class AudioSoundWavesEffectNode:
    """
    Node này tạo hiệu ứng "sound waves" dựa trên audio và dán lên video.
    Xuất ra 2 output: đường dẫn video (STRING) và video (VIDEO).
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Định nghĩa input cho node
        """
        return {
            "required": {
                "audio_path": ("STRING", {"default": "path/to/audio.wav"}),
                "video_path": ("STRING", {"default": "path/to/video.mp4"}),
                "waves_type": (["bar", "wave"],),
                "position": (["center", "top_1_3", "bottom_1_3"],),
                "wave_height": ("INT", {"default": 100, "min": 10, "max": 2000, "step": 1}),
                "wave_width": ("INT", {"default": 800, "min": 10, "max": 4000, "step": 1}),
                "wave_color": ("STRING", {"default": "255,255,255"}),  # RGB
            }
        }

    # Khai báo kiểu dữ liệu trả về
    RETURN_TYPES = ("STRING", "VIDEO",)
    RETURN_NAMES = ("output_video_path", "processed_video",)
    FUNCTION = "apply_sound_waves"
    CATEGORY = "Video/Audio"

    def apply_sound_waves(self, audio_path, video_path, waves_type="bar", position="center",
                          wave_height=100, wave_width=800, wave_color="255,255,255"):
        """
        Đây là hàm chính được gọi bởi ComfyUI.  
        Sau khi xử lý, trả về (output_video_path, processed_video).
        """

        # ---------------------
        # 1) Thực hiện transform video + audio
        # ---------------------
        output_path = self.transform(
            audio_path=audio_path,
            video_path=video_path,
            waves_type=waves_type,
            position=position,
            wave_height=wave_height,
            wave_width=wave_width,
            wave_color=wave_color
        )

        # ---------------------
        # 2) Mở file video đã tạo, đọc data bytes để trả về VIDEO
        # ---------------------
        processed_video = None
        if os.path.exists(output_path):
            with open(output_path, 'rb') as f:
                video_bytes = f.read()
            processed_video = {
                "path": output_path,
                "data": video_bytes
            }
        else:
            raise FileNotFoundError(f"Video output not found at {output_path}")

        # ---------------------
        # 3) Trả về tuple 2 output
        # ---------------------
        return (output_path, processed_video)

    def transform(self, audio_path, video_path, waves_type,
                  position, wave_height, wave_width, wave_color):
        """
        Tạo video output_path chứa hiệu ứng waves dựa trên audio. 
        """
        # 1) Parse màu cho wave
        try:
            color_list = [int(c) for c in wave_color.split(",")]
            if len(color_list) != 3:
                color_list = [255, 255, 255]
        except:
            color_list = [255, 255, 255]

        # 2) Đọc audio, phân tích amplitude
        audio = AudioSegment.from_file(audio_path)
        sample_rate = audio.frame_rate
        audio_mono = audio.split_to_mono()[0]
        samples = np.array(audio_mono.get_array_of_samples())
        audio_duration = audio.duration_seconds

        # 3) Đọc video gốc
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 4) Tạo video writer
        output_name = self._generate_output_path(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

        # 5) Xử lý từng frame
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = frame_index / fps
            # Lấy sample index tương ứng
            sample_index = int(current_time * sample_rate)
            sample_index = max(0, min(sample_index, len(samples) - 1))

            amplitude = abs(samples[sample_index])
            amplitude_norm = amplitude / 32767.0

            # Vẽ sóng
            frame = self._draw_sound_wave(
                frame=frame,
                waves_type=waves_type,
                amplitude_norm=amplitude_norm,
                position=position,
                wave_height=wave_height,
                wave_width=wave_width,
                wave_color=color_list
            )

            out.write(frame)
            frame_index += 1

        cap.release()
        out.release()

        return output_name

    def _draw_sound_wave(self, frame, waves_type, amplitude_norm,
                         position, wave_height, wave_width, wave_color):
        """
        Vẽ hiệu ứng (bar hoặc wave) lên frame
        """
        h, w, _ = frame.shape
        # Xác định toạ độ y để vẽ (center, top_1_3, bottom_1_3)
        if position == "center":
            center_y = h // 2
        elif position == "top_1_3":
            center_y = h // 3
        elif position == "bottom_1_3":
            center_y = 2 * h // 3
        else:
            center_y = h // 2  # Mặc định

        # Giới hạn chiều rộng vẽ
        wave_width = min(wave_width, w)

        # Biên độ px cho sóng
        wave_amp = int(wave_height * amplitude_norm)

        # Xác định toạ độ x để vẽ
        start_x = (w - wave_width) // 2
        end_x = start_x + wave_width

        if waves_type == "bar":
            # Vẽ cột (bar)
            top_y = center_y - wave_amp
            bottom_y = center_y + wave_amp
            cv2.rectangle(frame, (start_x, top_y), (end_x, bottom_y), wave_color, -1)

        elif waves_type == "wave":
            # Vẽ đường sóng sin
            step = 2  # mỗi 2 px
            wave_points = []
            for x in range(start_x, end_x, step):
                progress = (x - start_x) / float(wave_width if wave_width else 1)
                offset = int(wave_amp * math.sin(2 * math.pi * progress))
                y = center_y - offset
                wave_points.append((x, y))
            wave_points = np.array(wave_points, np.int32)
            cv2.polylines(frame, [wave_points], False, wave_color, 2)
        else:
            # Các loại khác chưa hỗ trợ
            pass

        return frame

    def _generate_output_path(self, video_path):
        """
        Tạo output path, ví dụ: video.mp4 -> video_soundwaves.mp4
        """
        base, ext = os.path.splitext(video_path)
        return f"{base}_soundwaves{ext}"
    

# =============================================================================
# Đăng ký node vào ComfyUI
# =============================================================================
NODE_CLASS_MAPPINGS = {
    "AudioSoundWavesEffectNode": AudioSoundWavesEffectNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSoundWavesEffectNode": AudioSoundWavesEffectNode
}
