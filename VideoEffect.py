import os
import cv2
import numpy as np
from pydub import AudioSegment
import math

# Trong ComfyUI, ta thường import Node, NODE_CLASS_MAPPINGS, v.v... 
# từ các module sẵn có. Ví dụ:
try:
    from .. import nodes, util
except:
    # Nếu bạn để node này trong 1 file riêng chưa có module cha,
    # có thể bỏ qua hoặc tuỳ biến đường dẫn import
    pass

# =============================================================================
# Node AudioSoundWavesEffect
# =============================================================================
class AudioSoundWavesEffectNode:
    """
    Node này minh hoạ cách thêm hiệu ứng "sound waves" dựa trên audio vào video.
    Các tham số đầu vào (inputs) và đầu ra (outputs) được thiết kế cho ComfyUI.
    """
    def __init__(self):
        # Các thông tin cơ bản cho Node
        self.name = "Audio Sound Waves Effect"
        self.category = "Video/Audio"
        self.version = "1.0"

    @classmethod
    def INPUT_TYPES(cls):
        """
        Định nghĩa các input cho node. Ví dụ:
        - audio_path (str): Đường dẫn audio
        - video_path (str): Đường dẫn video gốc
        - waves_type (str): Loại sóng (cột hoặc sóng)
        - position (str): Vị trí hiển thị (center, top_1_3, bottom_1_3)
        - wave_height (int): Chiều cao wave (pixel)
        - wave_width (int): Chiều rộng wave (pixel)
        - wave_color (list): Màu RGB
        """
        return {
            "required": {
                "audio_path": ("STRING", {"default": "path/to/audio.wav"}),
                "video_path": ("STRING", {"default": "path/to/video.mp4"}),
                "waves_type": (["bar", "wave"],),
                "position": (["center", "top_1_3", "bottom_1_3"],),
                "wave_height": ("INT", {"default": 100, "min": 10, "max": 2000, "step": 1}),
                "wave_width": ("INT", {"default": 800, "min": 10, "max": 4000, "step": 1}),
                # Màu dạng [R, G, B], mỗi giá trị 0..255
                "wave_color": ("STRING", {"default": "255,255,255"}),
            }
        }

    @classmethod
    def OUTPUT_TYPES(cls):
        """
        Định nghĩa output cho node. Ở đây, ta sẽ xuất ra đường dẫn
        của video đã thêm hiệu ứng (hoặc handle nào đó).
        """
        return ("STRING",)

    @classmethod
    def IS_CHANGED(cls):
        """
        ComfyUI sử dụng cờ này để xác định xem Node có cần chạy lại hay không.
        Thường để True cho đơn giản.
        """
        return True

    def transform(self, audio_path, video_path, waves_type="bar", position="center",
                  wave_height=100, wave_width=800, wave_color="255,255,255"):
        """
        Hàm chính để:
        1) Đọc dữ liệu audio (amplitude) theo thời gian.
        2) Đọc video theo từng frame.
        3) Vẽ hiệu ứng sóng âm lên frame.
        4) Xuất video có chứa hiệu ứng.
        """

        # Parse wave_color dạng "R,G,B"
        color_list = [int(c) for c in wave_color.split(",")]
        if len(color_list) != 3:
            color_list = [255, 255, 255]  # nếu parse sai thì default trắng

        # 1) Đọc audio, phân tích amplitude
        audio = AudioSegment.from_file(audio_path)
        # Lấy số sample mỗi giây (tần số lấy mẫu)
        sample_rate = audio.frame_rate
        # Dữ liệu raw (mono hoặc stereo), để đơn giản ta chuyển về mono
        audio_mono = audio.split_to_mono()[0]
        # Lấy mảng sample (16-bit)
        samples = np.array(audio_mono.get_array_of_samples())

        # Để có được amplitude cho từng frame, 
        # ta cần biết audio dài bao nhiêu => audio.duration_seconds
        audio_duration = audio.duration_seconds

        # 2) Đọc video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video {video_path}")
        
        # Lấy thông tin video (fps, size)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Tạo video writer để ghi video đầu ra
        output_name = self._generate_output_path(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Hoặc 'avc1', 'XVID', ...
        out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

        # 3) Xử lý từng frame
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Thời điểm (giây) tương ứng với frame này
            current_time = (frame_index / fps)

            # Tính chỉ số sample tương ứng
            sample_index = int(current_time * sample_rate)
            if sample_index < 0:
                sample_index = 0
            if sample_index >= len(samples):
                sample_index = len(samples) - 1

            # amplitude [-32768..32767], ta lấy giá trị tuyệt đối để vẽ
            amplitude = abs(samples[sample_index])
            # Chuẩn hoá amplitude về [0..1] đơn giản
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
        Vẽ hiệu ứng sóng (bar hoặc wave) lên frame.
        """

        h, w, _ = frame.shape
        # Tính toạ độ gốc (anchor) để vẽ
        if position == "center":
            center_y = h // 2
        elif position == "top_1_3":
            center_y = h // 3
        elif position == "bottom_1_3":
            center_y = (2 * h) // 3
        else:
            center_y = h // 2  # default

        # Giới hạn chiều rộng wave
        wave_width = min(wave_width, w)

        # Biên độ (pixel) cho sóng so với amplitude_norm
        # amplitude_norm ~ [0..1], wave_height là max pixel
        wave_amp = int(wave_height * amplitude_norm)

        # Tính toạ độ bắt đầu - kết thúc theo chiều ngang (để vẽ)
        start_x = (w - wave_width) // 2
        end_x = start_x + wave_width

        if waves_type == "bar":
            # Dạng cột (bar) sẽ vẽ 1 hình chữ nhật
            top_y = center_y - wave_amp
            bottom_y = center_y + wave_amp
            cv2.rectangle(frame, (start_x, top_y), (end_x, bottom_y), wave_color, -1)

        elif waves_type == "wave":
            # Dạng sóng (răng cưa đơn giản), ta vẽ 1 polyline
            # Mỗi x step khoảng 2 pixel
            step = 2
            wave_points = []
            for x in range(start_x, end_x, step):
                # progress [0..1] tính từ start_x -> end_x
                progress = (x - start_x) / (wave_width if wave_width != 0 else 1)
                # Tạo dao động kiểu sin
                offset = int(wave_amp * math.sin(2 * math.pi * progress))
                y = center_y - offset
                wave_points.append((x, y))

            wave_points = np.array(wave_points, np.int32)
            cv2.polylines(frame, [wave_points], False, wave_color, 2)

        else:
            # Mặc định, nếu chưa hỗ trợ, không vẽ gì.
            pass

        return frame

    def _generate_output_path(self, video_path):
        """
        Tạo đường dẫn output (để không ghi đè file gốc).
        VD: video.mp4 -> video_soundwaves.mp4
        """
        base, ext = os.path.splitext(video_path)
        return f"{base}_soundwaves{ext}"

    def __call__(self, inputs):
        """
        Hàm __call__ được ComfyUI gọi khi node được chạy.
        inputs là dict chứa các key tương ứng INPUT_TYPES.
        Ta parse ra và gọi transform.
        Sau khi xử lý, return (output_path,) (tuple)
        """
        audio_path = inputs["audio_path"]
        video_path = inputs["video_path"]
        waves_type = inputs["waves_type"]
        position = inputs["position"]
        wave_height = inputs["wave_height"]
        wave_width = inputs["wave_width"]
        wave_color = inputs["wave_color"]

        output_path = self.transform(
            audio_path=audio_path,
            video_path=video_path,
            waves_type=waves_type,
            position=position,
            wave_height=wave_height,
            wave_width=wave_width,
            wave_color=wave_color
        )
        return (output_path,)
    

# =============================================================================
# Đăng ký node vào ComfyUI
# =============================================================================
NODE_CLASS_MAPPINGS = {
    "AudioSoundWavesEffectNode": AudioSoundWavesEffectNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSoundWavesEffectNode": AudioSoundWavesEffectNode
}
