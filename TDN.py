import librosa
import numpy as np
import matplotlib.pyplot as plt
from music21 import stream, note, chord, tempo, meter

# 定义调式音符
def get_scale(key):
    scales = {
      # 大调 (Major Scales)
    'C major': ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    'C# major': ['C#', 'D#', 'E#', 'F#', 'G#', 'A#', 'B#'],
    'D major': ['D', 'E', 'F#', 'G', 'A', 'B', 'C#'],
    'E- major': ['E-', 'F', 'G', 'A-', 'B-', 'C', 'D'],
    'E major': ['E', 'F#', 'G#', 'A', 'B', 'C#', 'D#'],
    'F major': ['F', 'G', 'A', 'B-', 'C', 'D', 'E'],
    'F# major': ['F#', 'G#', 'A#', 'B', 'C#', 'D#', 'E#'],
    'G major': ['G', 'A', 'B', 'C', 'D', 'E', 'F#'],
    'A- major': ['A-', 'B-', 'C', 'D-', 'E-', 'F', 'G'],
    'A major': ['A', 'B', 'C#', 'D', 'E', 'F#', 'G#'],
    'B- major': ['B-', 'C', 'D', 'E-', 'F', 'G', 'A'],
    'B major': ['B', 'C#', 'D#', 'E', 'F#', 'G#', 'A#'],

    # 小调 (Minor Scales)
    'C minor': ['C', 'D', 'E-', 'F', 'G', 'A-', 'B-'],
    'C# minor': ['C#', 'D#', 'E', 'F#', 'G#', 'A', 'B'],
    'D minor': ['D', 'E', 'F', 'G', 'A', 'B-', 'C'],
    'E- minor': ['E-', 'F', 'G-', 'A-', 'B-', 'C-', 'D-'],
    'E minor': ['E', 'F#', 'G', 'A', 'B', 'C', 'D'],
    'F minor': ['F', 'G', 'A-', 'B-', 'C', 'D-', 'E-'],
    'F# minor': ['F#', 'G#', 'A', 'B', 'C#', 'D', 'E'],
    'G minor': ['G', 'A', 'B-', 'C', 'D', 'E-', 'F'],
    'A- minor': ['A-', 'B-', 'C-', 'D-', 'E-', 'F-', 'G-'],
    'A minor': ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    'B- minor': ['B-', 'C', 'D-', 'E-', 'F-', 'G-', 'A-'],
    'B minor': ['B', 'C#', 'D', 'E', 'F#', 'G', 'A'],

    # default 调式（涵盖所有音符）
    'default': ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
                'B-', 'A-', 'G-', 'F-', 'E-', 'D-', 'C-', 'B#', 'E#', 'F#', 'A#']
        # 添加其他调式
    }
    return scales.get(key, [])
# 2. 和弦检测（使用librosa的音高检测功能）
def detect_chords(y, sr, hop_length=512, skip_factor=4, amplitude_threshold=3):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, hop_length=hop_length)
    chord_values = []
    time_steps = []

    # 通过跳过一些帧来降低采样频率
    for t in range(0, pitches.shape[1], skip_factor):
        # 获取幅度超过阈值的所有音高
        pitch_indices = np.where(magnitudes[:, t] > amplitude_threshold)[0]
        current_chord = []

        for index in pitch_indices:
            pitch = pitches[index, t]
            if pitch > 0:  # 只保留有效音高
                current_chord.append(pitch)
        
        if current_chord:
            chord_values.append(current_chord)
            time_steps.append(t * hop_length / sr)  # 将帧转换为时间
        else:
            chord_values.append(None)  # None表示休止符
            time_steps.append(t * hop_length / sr)

    return chord_values, time_steps

# 查找最相似音符的辅助函数
def find_closest_note(n, key_notes):
    input_note = note.Note(n)
    input_midi = input_note.pitch.midi

    closest_note = None
    closest_distance = float('inf')

    for key_note_name in key_notes:
        key_note = note.Note(key_note_name)
        key_midi = key_note.pitch.midi

        distance = abs(input_midi - key_midi)

        if distance < closest_distance:
            closest_distance = distance
            closest_note = key_note_name

    return closest_note

# 3. 和弦转换（将音高转换为和弦）
def chords_to_notes(chords, key_notes):
    notes = []
    last_chord = chord.Chord(None)

    for chord_pitches in chords:
        if chord_pitches is None:  # 休止符
            n = note.Rest(quarterLength=1/4)
            notes.append(n)
            last_chord = chord.Chord(None)
        else:
            chord_notes = []
            for pitch in chord_pitches:
                midi_note = librosa.hz_to_midi(pitch)
                n = note.Note()
                n.pitch.midi = int(round(midi_note))

                if n.name not in key_notes:
                    closest_note = find_closest_note(n.name, key_notes)
                    
                    n.name = closest_note
                
                chord_notes.append(n)

            # 生成和弦
            if chord_notes:
                c = chord.Chord(chord_notes)
                c.quarterLength = 1/4

                # 检查是否为相同和弦
                if last_chord.pitchNames == c.pitchNames:
                    c.quarterLength += last_chord.quarterLength
                    notes.pop()
                notes.append(c)
                last_chord = c
    return notes

# 4. 生成五线谱
def generate_score(chords, key_notes):
    s = stream.Score()
    p = stream.Part()

    # 设置拍号和节奏
    ts = meter.TimeSignature('4/4')
    p.append(ts)
    met = tempo.MetronomeMark(number=120)
    p.append(met)

    # 将和弦添加到乐谱
    notes = chords_to_notes(chords, key_notes)
    for n in notes:
        p.append(n)

    s.insert(0, p)
    return s

# 5. 可视化音频和和弦
def plot_audio_with_pitches(y, sr, time_steps, chords):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # 第一幅图：绘制音频波形
    librosa.display.waveshow(y, sr=sr, ax=ax1)

    for t, chord_pitches in zip(time_steps, chords):
        if chord_pitches is None:
            ax1.axvline(x=t, color='b', linestyle='--')
        else:
            ax1.axvline(x=t, color='r', linestyle='--')

    ax1.set_title('Audio Waveform with Detected Chord Lines')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)

    # 第二幅图：根据和弦音高绘制音轨
    midi_chords = [[librosa.hz_to_midi(pitch) for pitch in chord_pitches] if chord_pitches is not None else [np.nan] for chord_pitches in chords]
    avg_midi_pitches = [np.nanmean(chord) if chord is not None else np.nan for chord in midi_chords]

    ax2.plot(time_steps, avg_midi_pitches, linestyle='-', color='green', linewidth=3)

    ax2.set_title('Pitch Track')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Average MIDI Note Number')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()

# 主程序
def audio_to_sheet(audio_path, key, skip_factor=4,amplitude_threshold=3):
    y, sr = librosa.load(audio_path)
    key_notes = get_scale(key)

    chords, time_steps = detect_chords(y, sr, skip_factor=skip_factor,amplitude_threshold=amplitude_threshold)

    plot_audio_with_pitches(y, sr, time_steps, chords)

    score = generate_score(chords, key_notes)
    return score

# 5. 保存五线谱为MIDI或者MusicXML格式
def save_score(score, filename):
    score.write('musicxml', fp=filename)

# 示例：将哼唱的音频转换为五线谱
audio_path = 'Mozart_Cmajor.wav'
key = input("请输入调式（如 'C minor'）：")
amplitude_threshold = int(input("请输入噪声强度（如 '30'）："))
score = audio_to_sheet(audio_path, key, skip_factor=6,amplitude_threshold=amplitude_threshold)
save_score(score, 'output_score.xml')