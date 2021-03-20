import sys
import getopt
from pytube import YouTube

def usage():
    print("Youtube tool")
    print()
    print("Usage download_video.py --url video_url")
    #print("Usage bhpnet.py -t target_host -p port")
    #print("-u --url                     - download on [host]:[port] for incoming connections")
    print("-u --url                - download a video from an URL")
    print("")
    print("")
    print("Examples: ")
    print("download_video.py -u https://www.youtube.com/watch?v=xDyBIpqZcNI")
    sys.exit(0)

def main():
    """
    The main method allows us to define the activity to do in our program
    """
    if not len(sys.argv[1:]):
        usage()
    try:
        opts, args = getopt.getopt(sys.argv[1:], "u", "url")
    except Exception as e:
        print(str(e))

    if ((opts[0][0] == '-u') | (opts[0][0] == '--url')):

        YouTube(args[0]).streams.first().download(output_path="./videos")
        # print(video.streams.filter(file_extension = "mp4").all())
        print("video downloaded")
    else:
        sys.exit(0)
    # settings()
    # out = True
    # while out:
    #     options = input("Write the number of the option:\n (1) - Capture the network traffic \n (2) - Dataset generator"
    #                     "\n (3) - Exit")
    #     if int(options) == 1:
    #         capture()
    #     if int(options) == 2:
    #         feature_generator()
    #     if int(options) == 3:
    #         out = False
    #         exit(0)


if __name__ == "__main__":
    main()