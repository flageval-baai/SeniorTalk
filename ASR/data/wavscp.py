import os

def generate_wav_scp(data_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    file_id = os.path.splitext(file)[0]
                    f.write(f"{file_id} {file_path}\n")

if __name__ == "__main__":
    data_dir = "/home/chenyang/chenyang_space/speech_editing_and_tts/database/Data_eldly/wav_split/test"
    output_file = "/home/chenyang/chenyang_space/speech_editing_and_tts/wenet/examples/aishell/s0/data/test/wav.scp"
    generate_wav_scp(data_dir, output_file)