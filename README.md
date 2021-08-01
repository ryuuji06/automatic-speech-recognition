# Character-wise Automatic Speech Recognition

In this test, I implemented a simple functional system that transcript the speech of a given audio signal, task known as automatic speech recognition (ASR). 

Traditionally solved with HMM. With deep neural networks, it has been solved with CTC (or RNN-transceiver),
encoder-decoder (with or without attention) and transformer. In this test, I used a RNN-CTC approach.


One difficulty faced when implemented such systems is the vocabulary size to be recognized. To simplify this issue,
the implemented system recognizes a sequence of characters (letters), rather than using words as recognition units,
as usually done. As well-known, in English, the pronounce of a letter varies a lot, specially vowels. However, we
expect that the sequential learn of RNN can incorporate some orthographical characteristics of the English language.


# About the Model

# Sample Results


## References



[1] A. Graves, S. Fernandez, F. Gomez and J. Schmidhuber. "Connectionist temporal classification: labelling
unsegmented sequence data with recurrent neural networks". Proceedings of the 23rd International Conference on 
Machine Learning (ICML'06), p.369-376, 2006

[] https://github.com/lucko515/speech-recognition-neural-network
