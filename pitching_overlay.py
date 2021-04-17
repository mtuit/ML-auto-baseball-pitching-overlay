import argparse
import tensorflow as tf

from tensorflow.python.saved_model import tag_constants
from pathlib import Path

from src.get_pitch_frames import get_pitch_frames
from src.generate_overlay import generate_overlay
from src.utils import get_project_root, initialize_tensorflow




def parse_args():
    parser = argparse.ArgumentParser(description="Generate pitching overlays from a pitching sequence.")

    parser.add_argument('-i', '--input_directory', default=Path('videos'),
                        help='Location to root directory which contains pitching videos. (Default: videos/)')
    parser.add_argument('-m', '--model_location', default=Path(get_project_root() / 'model' / 'yolov4-tiny-baseball-416'), 
                        help="Set the location of the model. (Default: $ROOT/model/yolov4-tiny-baseball-416")
    parser.add_argument('-v', '--verbose', default=0, type=int, choices=[0, 1],
                        help="Set the verbosity of the program, choices [0, 1]. (Default: 0)")

    parser.add_argument('--size', type=int, default=416, 
                        help="Set the size hyperparameter. (Default: 416)")
    parser.add_argument('--iou', type=float, default=0.45,
                        help="Set the iou hyperparameter. (Default: 0.45)")
    parser.add_argument('--score', type=float, default=0.5,
                        help="Set the score hyperparameter. (Default: 0.5)")

    return parser.parse_args()


def main():
    args = parse_args()
    logger = tf.get_logger()
    initialize_tensorflow()
    
    saved_model_loaded = tf.saved_model.load(str(args.model_location), tags=[tag_constants.SERVING])
    model = saved_model_loaded.signatures['serving_default']

    input_dir = args.input_directory # Input directory is the highest level, containing directories of pitch sequences
    all_videos = list(input_dir.iterdir())

    # Iterate all pitching sequences in the input directory
    for idx, video_dir in enumerate(all_videos):
        logger.info(f'[{idx+1}/{len(all_videos)+1}] Currently processing {video_dir}.') 
        output_path = Path(video_dir / f'{video_dir.stem}_overlay.mp4')

        if output_path.exists(): 
            logger.info(f'Overlay is already present in the directory, skipping {video_dir}...')
            continue

        pitch_frames = []

        # Iterate all pitches in the sequence directory
        for pitch_video in video_dir.iterdir(): 
            logger.info(f'Tracking baseball in {pitch_video}')
            try:
                ball_frames, width, height, fps = get_pitch_frames(str(pitch_video), model, args.size, args.iou, args.score, args.verbose)
                pitch_frames.append(ball_frames)
            except Exception as e:
                logger.error(f'Sorry we could not get enough baseball detection from the video, videos from {pitch_video} will not be overlayed:')
                logger.error(e)

        if (len(pitch_frames)):
            generate_overlay(pitch_frames, width, height, fps, str(output_path), args.verbose)
    
if __name__ == "__main__": 
    main()