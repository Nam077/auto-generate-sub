from faster_whisper import WhisperModel
import json
from pathlib import Path
import torch
import ass
import subprocess
import os
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich.panel import Panel
import numpy as np
import librosa

class MediaProcessor:
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma'}
    
    def __init__(self, model_size="large-v3", device=None, temp_dir="temp", output_dir="result"):
        self.console = Console()
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        compute_type = "float16" if device == "cuda" else "int8"
        
        with self.console.status(f"Loading {model_size} model on {device}...", spinner="dots"):
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
        
        self.temp_dir = Path(temp_dir)
        self.output_base_dir = Path(output_dir)
        
        self.temp_dir.mkdir(exist_ok=True)
        self.output_base_dir.mkdir(exist_ok=True)
        self.check_ffmpeg()
        
        # Initialize a single Progress instance
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )

    def validate_audio(self, audio_path):
        """Validate audio file without checking SNR."""
        try:
            y, sr = librosa.load(audio_path, sr=None, duration=10)
            
            # Kiểm tra xem file có quá im lặng không
            if np.max(np.abs(y)) < 0.01:
                return False, "Audio appears to be silent or extremely quiet"
            
            # Kiểm tra tần số mẫu của file
            if sr < 8000:
                return False, "Sample rate too low for reliable transcription"
            
            # Kiểm tra độ dài của audio
            duration = librosa.get_duration(y=y, sr=sr)
            if duration < 0.1:
                return False, "Audio file too short"
            
            # Kiểm tra hiện tượng clipping
            if np.any(np.abs(y) >= 1.0):
                self.console.print("[yellow]Warning: Audio contains clipping[/yellow]")
            
            # Bỏ qua kiểm tra SNR, luôn trả về thành công
            return True, "Audio validation passed"
    
        except Exception as e:
            return False, f"Audio validation failed: {str(e)}"


    def check_ffmpeg(self):
        """Check if ffmpeg is installed"""
        with self.console.status("Checking FFmpeg installation...", spinner="dots"):
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                self.console.print("[green]FFmpeg check passed[/green]")
                return True
            except (subprocess.CalledProcessError, FileNotFoundError):
                self.console.print("[red]FFmpeg is not installed or not found in PATH[/red]")
                raise RuntimeError("FFmpeg is not installed or not found in PATH")

    def get_media_info(self, file_path):
        """Get media file information using ffprobe"""
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            '-show_streams',
            str(file_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, check=True, text=True)
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            self.console.print(f"[red]Error getting media info: {e.stderr}[/red]")
            raise

    def convert_to_mp3(self, input_path, bitrate='192k', task_id=None):
        """Convert any supported media file to MP3 in temp directory"""
        input_path = Path(input_path)
        output_path = self.temp_dir / f"{input_path.stem}_temp.mp3"
        
        if task_id is None:
            task_id = self.progress.add_task(f"Converting {input_path.name} to MP3...", total=100)
            
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-vn',
            '-acodec', 'libmp3lame',
            '-ab', bitrate,
            '-ar', '44100',
            '-y',
            str(output_path)
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            duration = float(self.get_media_info(input_path)['format']['duration'])
            
            while True:
                line = process.stderr.readline()
                if not line:
                    break
                
                if "time=" in line:
                    time_str = line.split("time=")[1].split()[0]
                    current_time = sum(float(x) * 60 ** i for i, x in enumerate(reversed(time_str.split(":"))))
                    self.progress.update(task_id, completed=(current_time/duration)*100)
            
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd)
            
            self.progress.update(task_id, completed=100)
            return output_path
            
        except subprocess.CalledProcessError as e:
            self.console.print(f"[red]Error converting file: {e.stderr}[/red]")
            raise

    def create_output_directory(self, input_file, custom_output_dir=None):
        """Create timestamped output directory for results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_path = Path(input_file)
        
        if custom_output_dir:
            base_dir = Path(custom_output_dir)
        else:
            base_dir = self.output_base_dir
            
        output_dir = base_dir / f"{input_path.stem}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return output_dir

    def process_media(self, file_path, language=None, keep_temp=False, output_dir=None):
        """Process any supported media file and extract lyrics"""
        with self.progress:
            file_path = Path(file_path)
            
            validation_task = self.progress.add_task("Validating file...", total=100)
            ext = file_path.suffix.lower()
            if ext not in self.SUPPORTED_VIDEO_FORMATS | self.SUPPORTED_AUDIO_FORMATS:
                raise ValueError(f"Unsupported file format: {ext}")
            
            media_info = self.get_media_info(file_path)
            
            has_audio = any(stream['codec_type'] == 'audio' 
                           for stream in media_info.get('streams', []))
            if not has_audio:
                raise ValueError("No audio stream found in the file")
            
            self.progress.update(validation_task, completed=100)

            try:
                if ext != '.mp3':
                    conversion_task = self.progress.add_task("Converting to MP3...", total=100)
                    mp3_path = self.convert_to_mp3(file_path, task_id=conversion_task)
                else:
                    mp3_path = file_path

                validation_task = self.progress.add_task("Validating audio...", total=100)
                status, message = self.validate_audio(mp3_path)
                if not status:
                    self.console.print(f"[red]Audio validation failed: {message}[/red]")
                    raise ValueError(message)
                self.progress.update(validation_task, completed=100)

                transcription_task = self.progress.add_task("Transcribing audio...", total=100)
                lyrics = self.extract_lyrics(mp3_path, language, transcription_task)
                
                output_path = self.create_output_directory(file_path, output_dir)
                
                saving_task = self.progress.add_task("Saving outputs...", total=100)
                self.save_lyrics(lyrics, output_path, saving_task)
                
                if not keep_temp and ext != '.mp3':
                    os.remove(mp3_path)
                    
                return lyrics, output_path
                
            except Exception as e:
                if not keep_temp and ext != '.mp3' and 'mp3_path' in locals():
                    if os.path.exists(mp3_path):
                        os.remove(mp3_path)
                raise e

    def extract_lyrics(self, audio_path, language=None, task_id=None):
        """Extract lyrics from audio file using Whisper"""
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            temperature=[0.1, 0.2],
            vad_parameters=dict(min_silence_duration_ms=100)
        )

        processed_segments = []
        segments = list(segments)
        total_segments = len(segments)
        
        for i, segment in enumerate(segments):
            words = []
            for word in segment.words:
                words.append({
                    "startTime": int(word.start * 1000),
                    "endTime": int(word.end * 1000),
                    "data": word.word
                })
            
            processed_segment = {
                "words": words,
                "startTime": int(segment.start * 1000),
                "endTime": int(segment.end * 1000),
                "rawText": segment.text.strip()
            }
            processed_segments.append(processed_segment)
            
            if task_id is not None:
                self.progress.update(task_id, completed=(i + 1) * 100 / total_segments)
        
        if task_id is not None:
            self.progress.update(task_id, completed=100)
        
        return {
            "segments": processed_segments,
            "language": info.language,
            "languageProbability": info.language_probability
        }

    def save_lyrics(self, lyrics, output_dir, task_id=None, formats=None):
        """Save lyrics in various formats to specified directory"""
        if formats is None:
            formats = ['json', 'txt', 'srt', 'ass']
            
        base_path = output_dir / "transcript"
        total_formats = len(formats)
        current_format = 0
        
        # Save JSON format
        if 'json' in formats:
            output_data = {
                "err": 0,
                "msg": "Success",
                "data": {
                    "sentences": lyrics["segments"]
                }
            }
            
            with open(f"{base_path}.json", 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            current_format += 1
            if task_id is not None:
                self.progress.update(task_id, completed=(current_format * 100 / total_formats))

        # Save TXT formats
        if 'txt' in formats:
            self._save_txt_formats(lyrics, base_path)
            current_format += 1
            if task_id is not None:
                self.progress.update(task_id, completed=(current_format * 100 / total_formats))

        # Save SRT format
        if 'srt' in formats:
            self._save_srt_format(lyrics, base_path)
            current_format += 1
            if task_id is not None:
                self.progress.update(task_id, completed=(current_format * 100 / total_formats))

        # Save ASS format
        if 'ass' in formats:
            self._save_ass_format(lyrics, base_path)
            current_format += 1
            if task_id is not None:
                self.progress.update(task_id, completed=(current_format * 100 / total_formats))

        self.console.print(f"[green]All files saved to: {output_dir}[/green]")
        return output_dir

    def _save_txt_formats(self, lyrics, base_path):
        # Detailed version
        with open(f"{base_path}_detailed.txt", 'w', encoding='utf-8') as f:
            f.write(f"Detected language: {lyrics['language']} (probability: {lyrics['languageProbability']:.2%})\n\n")
            for i, segment in enumerate(lyrics['segments'], 1):
                f.write(f"\nSegment {i}:\n")
                f.write(f"[{self.format_timestamp(segment['startTime'])} --> {self.format_timestamp(segment['endTime'])}]\n")
                f.write(f"Text: {segment['rawText']}\n")
                if segment['words']:
                    f.write("Words:\n")
                    for word in segment['words']:
                        f.write(f"  {word['data']}: {self.format_timestamp(word['startTime'])} --> {self.format_timestamp(word['endTime'])}\n")
                f.write("-" * 80 + "\n")
        
        # Simple version
        with open(f"{base_path}.txt", 'w', encoding='utf-8') as f:
            for segment in lyrics['segments']:
                f.write(f"[{self.format_timestamp(segment['startTime'])}] {segment['rawText']}\n")

    def _save_srt_format(self, lyrics, base_path):
        with open(f"{base_path}.srt", 'w', encoding='utf-8') as f:
            for i, segment in enumerate(lyrics['segments'], 1):
                f.write(f"{i}\n")
                f.write(f"{self.format_timestamp(segment['startTime'], 'srt')} --> {self.format_timestamp(segment['endTime'], 'srt')}\n")
                f.write(f"{segment['rawText']}\n\n")

    def _save_ass_format(self, lyrics, base_path):
        style = ass.Style(
            name='Default',
            fontname='Arial',
            fontsize=20,
            primary_colour="&H00FFFFFF",
            secondary_colour="&H0000FF00",
            outline_colour="&H00000000",
            back_colour="&H00000000",
            bold=True,
            italic=False,
            underline=False,
            strike_out=False,
            scale_x=100,
            scale_y=100,
            spacing=0,
            angle=0,
            border_style=1,
            outline=2,
            shadow=2,
            alignment=2,
            margin_l=10,
            margin_r=10,
            margin_v=10,
            encoding=0
        )

        doc = ass.Document()
        doc.styles.append(style)
        doc.info['Title'] = 'Lyrics'
        doc.info['ScriptType'] = 'v4.00+'
        
        for segment in lyrics['segments']:
            start = self.format_timestamp(segment['startTime'], 'ass')
            end = self.format_timestamp(segment['endTime'], 'ass')
            
            event = ass.Dialogue(
                start=start,
                end=end,
                style='Default',
                text=segment['rawText']
            )
            doc.events.append(event)

        with open(f"{base_path}.ass", 'w', encoding='utf-8') as f:
            doc.dump_file(f)

    def format_timestamp(self, ms, format='srt'):
        """Convert milliseconds to various timestamp formats"""
        seconds = ms / 1000
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        
        if format == 'srt':
            return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{int((seconds % 1) * 1000):03d}"
        elif format == 'ass':
            return f"{hours:01d}:{minutes:02d}:{seconds:05.2f}"
        else:
            return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{int((seconds % 1) * 1000):03d}"

def main():
    console = Console()

    try:
        processor = MediaProcessor(
            model_size="medium",
            output_dir="result",
            temp_dir="temp"
        )
        
        file_path = "microchip.mp4"
        
        lyrics, output_dir = processor.process_media(
            file_path,
            # language="vi",
            keep_temp=False,
        )
        
        console.print(Panel.fit(
            f"[green]Processing complete![/green]\nResults saved in: {output_dir}",
            title="Success"
        ))
        
    except Exception as e:
        console.print(Panel.fit(
            f"[red]Error processing file:[/red]\n{str(e)}",
            title="Error",
            border_style="red"
        ))

if __name__ == "__main__":
    main()