#!/usr/bin/env python3
"""Generate the compact YOLOX post-processing model used by the sample.

The model replaces an interpreted 8,400-row decode plus two full 8,400 x 80
sort outputs. It computes the best class for each anchor, decodes boxes, and
keeps only the strongest candidates before OpenMR's CPU NMS operator runs.
"""

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


ANCHOR_COUNT = 8400
CLASS_COUNT = 80
PRE_NMS_CANDIDATES = 128


def make_grid_and_stride():
    grids = []
    strides = []
    for grid_size, stride in ((80, 8), (40, 16), (20, 32)):
        for index in range(grid_size * grid_size):
            grids.append((index % grid_size, index // grid_size))
            strides.append((stride, stride))
    return np.asarray(grids, dtype=np.float32), np.asarray(strides, dtype=np.float32)


class YoloXPostprocess(tf.Module):
    def __init__(self):
        super().__init__()
        grid, stride = make_grid_and_stride()
        self.grid = tf.constant(grid)
        self.stride = tf.constant(stride)

    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[1, ANCHOR_COUNT, 5 + CLASS_COUNT], dtype=tf.float32, name="predictions")
        ]
    )
    def __call__(self, predictions):
        predictions = predictions[0]
        raw_boxes = predictions[:, :4]
        objectness = predictions[:, 4]
        class_probabilities = predictions[:, 5:]

        class_ids = tf.argmax(class_probabilities, axis=1, output_type=tf.int32)
        class_scores = tf.reduce_max(class_probabilities, axis=1)
        scores = objectness * class_scores

        centers = (raw_boxes[:, :2] + self.grid) * self.stride
        sizes = tf.exp(raw_boxes[:, 2:]) * self.stride
        boxes = tf.concat((centers - sizes * 0.5, centers + sizes * 0.5), axis=1)

        candidate_scores, candidate_indices = tf.math.top_k(
            scores, k=PRE_NMS_CANDIDATES, sorted=True
        )
        candidate_boxes = tf.gather(boxes, candidate_indices)
        candidate_classes = tf.gather(class_ids, candidate_indices)

        return {
            "candidate_boxes": tf.identity(candidate_boxes, name="candidate_boxes"),
            "candidate_classes": tf.identity(
                tf.reshape(candidate_classes, [PRE_NMS_CANDIDATES, 1]),
                name="candidate_classes",
            ),
            "candidate_scores": tf.identity(
                tf.reshape(candidate_scores, [PRE_NMS_CANDIDATES, 1]),
                name="candidate_scores",
            ),
        }


def main():
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("yolox_postprocess.tflite")
    module = YoloXPostprocess()
    concrete = module.__call__.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(model)

    interpreter = tf.lite.Interpreter(model_content=model)
    signature = interpreter.get_signature_list()
    print(f"wrote {output_path} ({len(model)} bytes)")
    print(f"signatures: {signature}")
    print("inputs:", [(item["name"], item["shape"].tolist(), str(item["dtype"])) for item in interpreter.get_input_details()])
    print("outputs:", [(item["name"], item["shape"].tolist(), str(item["dtype"])) for item in interpreter.get_output_details()])


if __name__ == "__main__":
    main()
