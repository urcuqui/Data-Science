import sys
import getopt
import speech_recognition as sr
import os
import moviepy.editor as me
from pandas import Series

"""
tendrá la responsabilidad de procesar un video identificado por
video_id en la carpeta de ./videos (si hay un mismo video_id con distintas extensiones,
escoger el mas reciente) y retornar los top n fragmentos de texto que más aparecen en
el video. El criterio de ordenamiento es el número de frames que el texto aparece en el
video y n es un argumento de entrada.

pip install pyaudio
pip install SpeechRecognition
conda install -c conda-forge speechrecognition
conda install -c conda-forge moviepy
"""

OUTPUT_AUDIO_FILE = "converted.wav"
OUTPUT_TEXT_FILE = "recognized.txt"


def usage():
    print("Recognizer tool")
    print()
    print("Usage python recognize_text.py --vid video_id --top n")
    print("-v --vid                - select a video file using its path")
    print("-t --top                - the first frames")
    print("")
    print("")
    print("Examples: ")
    print("recognize_text.py --vid https://www.youtube.com/watch?v=xDyBIpqZcNI --top 2")
    sys.exit(0)


def main():
    """
    The main method allows us to define the activity to do in our program
    """
    process = False
    if not len(sys.argv[1:]):
        usage()
    try:
        opts, args = getopt.getopt(sys.argv[1:], "v:t", ["vid", "top"])
        if (opts[0][0] == '-v') | (opts[0][0] == '--vid'):
            VIDEO_FILE = opts[0][1]
            if (opts[1][0] == '-t') | (opts[1][0] == '--top'):
                ti = int(args[0])
                process = True
        if process:
            video_clip = me.VideoFileClip(r"{}".format(VIDEO_FILE))
            video_clip.audio.write_audiofile(r"{}".format(OUTPUT_AUDIO_FILE))
            recognizer = sr.Recognizer()
            audio_clip = sr.AudioFile("{}".format(OUTPUT_AUDIO_FILE))
            with audio_clip as source:
                audio_file = recognizer.record(source, duration=ti)
            print("Please wait ...")
            result = recognizer.recognize_google(audio_file)
            result_dict = Series(result.split(" ")).value_counts().to_dict()
            with open(OUTPUT_TEXT_FILE, 'w') as file:
                for i in result_dict:
                    file.writelines(str(i) + ":"+ str(result_dict[i])+"\n")
            os.remove("converted.wav")
            print("The result is now available in {}".format(OUTPUT_TEXT_FILE))
        else:
            sys.exit(0)
    except Exception as e:
        print(str(e))
        sys.exit(0)


if __name__ == "__main__":
    main()