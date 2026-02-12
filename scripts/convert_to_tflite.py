"""Convert the saved Keras model to TensorFlow Lite format.

Usage examples:
  python scripts/convert_to_tflite.py --output model/tflite/best.tflite
  python scripts/convert_to_tflite.py --quantize dynamic

Options:
  --model PATH        Path to Keras model (defaults to model/model_config.find_model_file())
  --output PATH       Output .tflite path (defaults to model_config.TFLITE_PATH)
  --quantize TYPE     None|dynamic|float16|int8
  --representative DATA_DIR  Directory of images to use for representative dataset (required for int8)

Note: Full integer quantization requires a representative dataset function.
"""
import argparse
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from model import model_config


def representative_data_gen_from_dir(img_dir: Path, input_size=(224, 224), max_samples=100):
    from PIL import Image
    import numpy as np
    count = 0
    for p in img_dir.iterdir():
        if p.suffix.lower() not in ('.png', '.jpg', '.jpeg', '.bmp'):
            continue
        img = Image.open(p).convert('RGB').resize(input_size)
        arr = (np.asarray(img).astype('float32') / 255.0)
        arr = arr[np.newaxis, ...]
        yield arr
        count += 1
        if count >= max_samples:
            break


def convert(model_path: Path, out_path: Path, quantize: str = None, rep_dir: Path = None):
    import tensorflow as tf

    print(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize is None or quantize.lower() == 'none':
        print('No quantization: exporting full float model')
    elif quantize == 'dynamic':
        print('Applying dynamic range quantization')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantize == 'float16':
        print('Applying float16 quantization')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantize == 'int8':
        if rep_dir is None:
            raise RuntimeError('int8 quantization requires a representative dataset directory')
        print('Applying full integer (int8) quantization')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        rep_gen = representative_data_gen_from_dir(rep_dir, input_size=model_config.MODEL_INPUT_SIZE)
        converter.representative_dataset = lambda: rep_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    else:
        raise ValueError('Unsupported quantization type')

    print(f'Converting to TFLite (quantize={quantize})...')
    tflite_model = converter.convert()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'wb') as fh:
        fh.write(tflite_model)
    print(f'Wrote TFLite model to {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--output', type=str, default=str(model_config.TFLITE_PATH))
    parser.add_argument('--quantize', type=str, default=None, choices=['none', 'dynamic', 'float16', 'int8'])
    parser.add_argument('--representative', type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else model_config.find_model_file()
    out_path = Path(args.output)
    quant = args.quantize if args.quantize != 'none' else None
    rep_dir = Path(args.representative) if args.representative else None

    convert(model_path, out_path, quantize=quant, rep_dir=rep_dir)


if __name__ == '__main__':
    main()
