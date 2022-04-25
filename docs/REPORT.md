# BirdClef 2022: An Audio Classification Odyssey

Daniel Kim
Jonathan Tan

## Abstract and Introduction

Our challenge was to take audio recordings of various bird species and classify them according to features we extracted from the data. This ability is particularly useful to conservationists because it allows them to identify and track species in any given area much more easily than traditional visual based methods. Setting up a field microphone and pressing 'record' is much more labor efficient than lugging people and equipment around trying to get a visual on the species in question.

In our case, the Kaggle competition we are basing our project off of concerns the efforts of Hawaii conservationists to track and monitor various endangered species of birds in and around military bases in the Oahu region. This is particularly difficult because the terrain surrounding the region is wild and overgrown, making it difficult for researchers to catalog the various species that call the wilderness home.

We implemented a convolutional neural network to take in our processed audio data and classify them into 15 different bird species. The original Kaggle dataset contains more than 14,000 individual recordings and 152 separate species; however our computing environment (Google Colab) could not handle batching data over 152 labels without either 1. running out of memory or 2. being prohibitively long in terms of processing time.

The results of our implementation were promising, with an overall best score of 98% accuracy after 10 epochs.

## Related Work

During the course of this project we consulted many different sources on audio classification and neural networks, spending a great deal of time consulting documentation and implementations of Convolutional Neural Networks to better understand the problem and the possible roads to solutions. Below is a list of some of the most relevant resources to our project:

1.  [https://www.kaggle.com/code/nilaychauhan/cornell-birdcall-audio-recognition-using-jax-flax](https://www.kaggle.com/code/nilaychauhan/cornell-birdcall-audio-recognition-using-jax-flax) The primary resource used to inform our implementation. It addresses use cases of both the JAX framework and its accompanying FLAX neural network library, both of which were central to our implementation. Being completely new to both frameworks, not to mention machine learning in general, our implementation is heavily based on this existing project, with changes accounting for our unique file structure and organization.
2.  [https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5) This was very informative in outlining methodology
3.  [https://jax.readthedocs.io/en/latest/](https://jax.readthedocs.io/en/latest/) [https://flax.readthedocs.io/en/latest/](https://flax.readthedocs.io/en/latest/) JAX and FLAX Documentation
4.  [https://github.com/google/flax](https://github.com/google/flax) FLAX Github Repository
5. [https://github.com/8bitmp3/JAX-Flax-Tutorial-Image-Classification-with-Linen](https://github.com/8bitmp3/JAX-Flax-Tutorial-Image-Classification-with-Linen) Another implementation of image classification with JAX and FLAX
6. [BirdCLEF2022 - Kaggle Competition ](https://www.kaggle.com/competitions/birdclef-2022/overview)
7. [https://www.kaggle.com/nilaychauhan/convert-cornell-birdcall-recognition-to-tfrecords](https://www.kaggle.com/nilaychauhan/convert-cornell-birdcall-recognition-to-tfrecords)  
8. [https://www.kaggle.com/servietsky/fast-import-audio-and-save-spectrograms/notebook](https://www.kaggle.com/servietsky/fast-import-audio-and-save-spectrograms/notebook)
9. [https://www.kaggle.com/dhananjay3/simple-pytorch-starter/notebook](https://www.kaggle.com/dhananjay3/simple-pytorch-starter/notebook)
(https://www.kaggle.com/code/nilaychauhan/cornell-birdcall-audio-recognition-using-jax-flax#Training)  
11. [https://github.com/google/flax/tree/main/examples/imagenet](https://github.com/google/flax/tree/main/examples/imagenet)
14. [https://gist.github.com/fedelebron/b7be87a4feb88786cc142ef99931ff06#file-dog-classifier-ipynb](https://gist.github.com/fedelebron/b7be87a4feb88786cc142ef99931ff06#file-dog-classifier-ipynb)  
15. [Stanford CS 231 Poster Project Guidlines](http://cs231n.stanford.edu/project.html)
16. [Dr. Monogioudis - Jax Project](https://pantelis.github.io/data-science/aiml-common/projects/jaxworld/jax.html)

Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep learning solution for avian diversity monitoring. Ecological Informatics, 61, 101236. [[Source](https://www.sciencedirect.com/science/article/pii/S1574954121000273)]

Doshi, Ketan. (2021). Audio Deep Learning Made Simple: Sound Classification, Step-by-Step. Medium. [[Source](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5.)]

In this project, we as students, were heavily inspired by the knowledge available to us in the above resources. Most audio learning examples and tutorials we performed relied on the librosa Python package, and utilized the technique of converting audio into spectrograms.

We realized that although charts, and images such as spectrograms were visual tools, we had to consider them as maps or methods to represent the entirety of data without the constraint of time.  It was then, we could apply traditional machine learning methods commonly found in image recognition.

## Data

We were given a total of 14852 .ogg audio files, divided into subdirectories according to 152 separate species. ".ogg" files are compressed audio files, much like .mp3s.

Preprocessing data and augmentation was done in 6 steps as follows upon dataloading as follows

1.  Ensure all audio files were stereo audio files. Some of the files were given as mono, so when passed into the utility function we made them stereo by duplicating the existing audio into the second channel  

2.  Pad or truncate each audio file so they are the same length  

3.  We augmented this data by shifting each truncated file by a random time unit. Audio was wrapped around when moved.  

4.  We then used the audio data to create Mel-Spectrograms from the given audio data  

We further augmented our data by introducing random time and fequency blockouts to increase the variability of our data

## Methods

We chose to implement a convolutional neural network based on our research on audio classification. In every instance we found, audio classification was reduced to a computer vision problem by virtue of mel-spectrogram conversions. By converting audio data to image data, our convolutional neural network was able to classify bird calls / songs by taking in mel-spectrograms as input, where each mel-band in the spectrogram corresponded to a feature. Mel-Spectrograms are a visual representation of data that maps frequencies and their intensities to the Mel Scale, which is a logarithmic transformation of frequencies that more closely depicts how sounds throughout the frequency spectrum are perceived by the human ear. Generally, humans are better able to detect lower frequencies than higher frequencies; as such, the Mel scale is much more granular at low frequencies than higher frequencies.

![Mel Scale.gif](https://www.sfu.ca/sonic-studio-webdav/handbook/Graphics/Mel.gif)

For example, if we consider a tone at 440 hz (which corresponds to A4 on the piano), the next octave up would be 880 hz (A5). The full range of 12 pitches (based on the western scale) present between octaves is located between 440 and 880 hz.

If we go up several octaves, A6 is located at 3520 hz, while A7 is located at 7040 hz. In this instance, the distance between each consecutive pitch is far lengthier than that of the lower octaves. While to the human ear each distance between pitches remains perceptually consistent, the distance in hz is not linear. Converting the frequency spectrum to the mel scale allows us to better visualize the entire range of human aural perception.

![eq](https://www.fabfilter.com/img/products/pro-q-3-screenshot.jpg)

In the above image we see a representation of the mel scale in an audio equalizer program. Notice the logarithmic progression of distance between frequencies as the frequency values increase on the bottom.

Here are some examples of Mel spectrograms we have developed:

A house sparrow call:
![house sparrow](https://github.com/JMortonTan/oahu-jaxbill/blob/main/imgs/housesparrow.png?raw=true)

A Hawaiian Osprey:

![osprey](https://github.com/JMortonTan/oahu-jaxbill/blob/main/imgs/osprey.png?raw=true)

The methods we used to implement the neural network and feed it data were fairly standard, differing only in that we used the new frameworks JAX and FLAX. We used PyTorch data loaders to prepare batches of data to introduce into the CPU’s RAM. We were not able to utilize TPUs as accelerators as suggested by JAX and FLAX within Kaggle, although we found a compromise using a GPU, useful when computing large multidimensional arrays (each mel spectrogram had a shape (636,128,1); multiply that by our batch size 32 and the scale of data being input into the architecture becomes very large).

For our learning algorithm, we utilized the VGG19 convolutional neural network architecture.  This was a neural network architecture that was designed for an image learning competition, that we adapted to learn from our Mel spectrograms.

In experimentation, we attempted to utilize the VGG19 network for varying epochs from 5-25, with up to 50 classes (or species) of birds.  The notebook will default to 5 species, with 10 epochs due to hardware limitations.

This is a graph charting the evolution of our VGG19 neural network:

![graph](https://github.com/JMortonTan/oahu-jaxbill/blob/main/imgs/finalgraph.png?raw=true)

We were very delighted to achieve 98.54% accuracy rating for our sample.

##  Experiments

Our initial goal of using the entire dataset (14852 files across 152 species) were dashed when faced with the realities of processing large data sets. On both Colab and Kaggle notebooks, performance lagged to an unusable degree when transferring data batches from CPU to GPU, with a single batch (out of 6) taking hours to complete. We downsized slowly, starting from 30 species, to eventually settling on 5. Below is a comparison of the number of species represented in a batch vs. the total time to complete a batch:

1.  152 (+7 hrs)
2.  30 (1.5 hrs)
3.  15 (2 hrs ) * because of random choosing of species, it is likely that some species were overrepresented in terms of total number of audio examples, thus contributing to increased batching times, even with a reduced number of species to sample
4.  5 (5 mins)

In the interest of time, we limited the number of epochs to 5. Originally, we specified a num_epochs parameter of 30 which was time-prohibitive. We scaled the number of epochs in accordance with the number of species represented in each batch, which cut down on overall processing time. However, the most significant factor in time-cost remains the batching step, as the time to process all epochs was proportional to the batching cost:

1.  Time to process 30 epochs: (n/a because we stopped the batching process before execution
2.  Time to process 20 epochs with 30 species (20 minutes)
3.  Time to process 10 epochs with 15 species (10 minutes)
4.  Time to process 10 epochs with 5 species (5:48)

## Conclusion

Our key findings show us that JAX and its derivative libraries are key instruments to accelerating and enabling data processing at large amounts.

While we were quite excited with the accuracy of our trained model, in future attempts we would very much like to include the secondary labels included within Cornell Lab of Ornithology.  The dataset features rich labelling of ancillary information such as whether the bird was in song or in call, or subspecies.  There was also other locational information that could potentially enrich and enhance the types of predictions a data model could make.

We were also dismayed by the sheer power, resources and time that our codebase required to develop and train the models.  We are very interested in continuing to optimize our solutions in order to develop a stronger model.

In the future, we also wanted to experiment with transformations of the audio clips themselves.  To improve our pre-processing regimen by analyzing each audio clip and isolating in more detail each portion of a bird call.  By analyzing these sub-segments, we believe we would be able to recognize in greater fidelity the species of birds through greater background noise and other disturbances.

Overall, this was a great challenge, and our team was deeply pressured to expand our horizons.  We are thankful to the authors of the resources, that we have referenced, NJIT and Dr. Monogioudis, and the CS301 class for having taken this journey alongside us.
