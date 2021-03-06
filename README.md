# Character-wise Automatic Speech Recognition

I implement a simple functional system that transcript the speech of a given audio signal, task known as automatic speech recognition (ASR). Instead of recognizing entire words, which would require a large output structure, or recognizing phonemes, which would require a training set with the transcripted phonemes (which is actually available, such as TIMIT), in this first scratch I opt to implement a simple character-wise speech recognition system. As well-known, in English, the pronounce of a letter varies a lot, specially vowels. However, we expect that the sequential learning of RNN can incorporate some orthographical characteristics of the English language.

The state-of-the-art of automatic speech recognition was held for a long time by methods based on hidden Markov models (HMM) [1], often with some neural network-hybrid strategy. With the emergence of deep learning, recurrent neural networks (RNN) and convolutional neura networks (CNN) came to prevail in solving ASR. Just to highlight a few approaches that has arisen,
 - there is the RNN-CTC algorithm [2] (and also its variant RNN-transceiver [3]), that performs an alignment the input sequence and the target sequence when computing the loss;
 - there is the encoder-decoder (or sequence-to-sequence) models, where an encoder processes the input sequence, and the decoder produces the output sequence from the encoder output; this model works better when encoder and decoder is intermediated by an attention mechanism [4,5];
 - more recently, end-to-end convolutional models have been showing promising results for ASR, such as Wav2letter [6] and Jasper [7];
 - also recently, transformers (attention-based models) have also shown good results in speech recognition task, such as with Conformer [8].

## About this implementation

I implemented a simple RNN-CTC model in this first scratch, based on [9], and trained it on the Librispeech dataset [10]. The model contains two LSTM layers, both with 128 units, followed by a dense output layer with 29 units and softmax activation (28 characters plus the CTC null element). No dropout or batch normalization was used.

## Sample Results

The training loss along the training process is shown below. The validation loss gets stuck too early while the training loss keeps on reducing. This looks like overfitting, but I also guess the model is still not complex enough. For a next improvement, I'll try to use dropout and normalization before increasing the model complexity and include convolutional layers.

<img src="https://github.com/ryuuji06/automatic-speech-recognition/blob/main/images/ex_hist.png" width="500">

In the following I present some prediction results with the trained network. Inputing the audio signal of the speech

`The music came nearer and he recalled the words. The words of Shelley's fragment upon the moon wandering companionless pale for weariness.`

As a greedy estimate (taking always the highest outputs at each time instant), I obtain

`te meusycame mearan he ro cal tho wirids the werds of shollys frit met to pon the mo laneer yng comepinilus pa fo wearing as`

This prediction is far from good, as expected from the high loss value, however the predicted letter sequence resembles somehow the original sentence. Also testing some beam search predictions, I obtain

`te meus y came mearan he ro cal tho wirids the werds of sholys frit met to pon the mon laner yng come pini lus pal fo wear ing as`

`te meus y came mearan he rof cal tho wirids the werds of sholys frit met to pon the mon laner yng come pini lus pal fo wear ing as`

`te meus y came mearan he ro cal tho wirids the werds of sholys frit met to pon the mon laner yng come pini lus pal fo wearing as`

`te meus y came mearan he rof cal tho wirids the werds of sholys frit met to pon the mon laner yng come pini lus pal fo wearing as`

`te meus y came mearan he ro cal tho wirids the werds of sholys frit met to pon the mon laner yng come pini lus pal fo wear ing as`


## References

[1] L.R. Bahl, F. Jelinek, and R.L. Mercer, "A maximum likelihood approach to continuous speech recognition". IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 5, pp. 179???190, 1983.

[2] A. Graves, S. Fernandez, F. Gomez and J. Schmidhuber. "Connectionist temporal classification: labelling unsegmented sequence data with recurrent neural networks". Proceedings of the 23rd International Conference on Machine Learning (ICML'06), p.369-376, 2006

[3] A. Graves, A. Mohamed, G. Hinton, "Speech recognition with deep recurrent neural networks". Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2013.

[4] J. Chorowski, D. Bahdanau, K. Cho and Y. Bengio. "Attention-based models for speech recognition". Proceedings of the 28th International Conference on Neural Information Processing Systems (NIPS'15), p. 577-585, 2015.

[5] W. Chan, N. Jaitly, Q. Le and O. Vinyals. "Listen, attend and spell: a neural network for large vocabulary conversational speech recognition". Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP'16), p. 4960-4964, 2016.

[6] R. Collobert, C. Puhrsch and G. Synnaeve. "Wav2letter: and end-to-end ConvNet-based speech recognition system". Proceedings of the 5th International Conference on Learning Representations, 2017

[7] J. Li, V. Lavrukhin, B. Ginsburg, et al. "Jasper: an end-to-end convolutional neural acoustic model". Proceedings of the Interspeech 2019, p. 71-75.

[8] A. Gulati, J. Qin, C. Chiu, W. Han, et al. "Conformer: convolution-augmented transformer for speech recognition". Proceedings of the Interspeech 2020.

[9] https://github.com/lucko515/speech-recognition-neural-network

[10] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur. "Librispeech: an ASR corpus based on public domain audio books". Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2015, p. 5206-5210.
