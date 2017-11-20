import tensorflow as tf
import numpy as np
import tempfile

tmp_filename = 'tf.tmp'

sequences = [[1, 2, 3], [1, 2], [3, 2, 1]]
label_sequences = [[0, 1, 0], [1, 0], [1, 1, 1]]

def make_example(input_sequence, output_sequence):
    """
    Makes a single example from Python lists that follows the
    format of tf.train.SequenceExample.
    """

    example_sequence = tf.train.SequenceExample()

    # 3D length
    sequence_length = len(input_sequence)

    example_sequence.context.feature["length"].int64_list.value.append(sequence_length)

    input_characters = example_sequence.feature_lists.feature_list["input_characters"]
    output_characters = example_sequence.feature_lists.feature_list["output_characters"]

    for input_character, output_character in zip(input_sequence,
                                                          output_sequence):

        if input_sequence is not None:
            input_characters.feature.add().int64_list.value.append(input_character)

        if output_characters is not None:
            output_characters.feature.add().int64_list.value.append(output_character)

    return example_sequence

# Write all examples into a TFRecords file
def save_tf(filename):
