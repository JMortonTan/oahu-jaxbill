# oahu-jaxbill

NJIT CS301-102 Group 15's course project repository.
Class taught by Dr. Pantelis Monogioudis, TA Yashwee Kothari.

A BirdCLEF 2022 Kaggle Competition undergraduate student entry.

![alt text](https://cdn.download.ams.birds.cornell.edu/api/v1/asset/78545161/1800)

|  | Links |
|--|--|
| Project Outline | **[https://pantelis.github.io/data-science/aiml-common/projects/jaxworld/jax.html#](https://pantelis.github.io/data-science/aiml-common/projects/jaxworld/jax.html#)** |
| Project Guidlines | **[https://pantelis.github.io/data-science/aiml-common/projects/_index.html](https://pantelis.github.io/data-science/aiml-common/projects/_index.html)** |
| BirdCLEF 2022 Kaggle | **[https://www.kaggle.com/competitions/birdclef-2022](https://www.kaggle.com/competitions/birdclef-2022)** |
| Evaluation Method | **[https://en.wikipedia.org/wiki/F-score](https://en.wikipedia.org/wiki/F-score)** |

## Team
|Name| Github |
|--|--|
| Daniel Kim | **[https://github.com/miknad2319](https://github.com/miknad2319)** |
| Jonathan Tan | **[https://github.com/JMortonTan](https://github.com/JMortonTan)** |

## Proposal

We are Bioimperium, a tech-solutions based environmental resource management group.

We’ve recently been awarded a contract by the State of Hawai'i to perform ecological management of native audubon populations across the island. Hawai'i’s large military presence has led to population disturbances for many of Hawai’i’s at-risk and endangered wildlife.

Base security, terrain access, and manpower are limitations for traditional solutions. We are deploying soundscape recorders in targeted areas that will utilize trained machine models to determine, discern, and count bird populations.

**What is the problem that you will be investigating? Why is it interesting?**

We are investigating the classification of bird species using audio recordings of bird songs. Being able to classify birds by sound takes less time and energy than visual methods (setting up and pressing record vs. hiking and maneuvering), which makes it very useful when it comes to monitoring more inaccessible areas of Hawaiian geography.

Our analysis of audio data must also account for confounding factors such as vocalizations of other non-avian species, sounds from military installations and aircraft, and ambient noise from the environment. This poses a challenge, especially when multiple sources of noise are present in a single audio file.

The successful implementation of our method would decrease equipment and manpower costs, allowing us to efficiently monitor and gather data from large areas in ways that traditional visual methods alone could not.

**What reading will you examine to provide context and background?**

Kahl, S., Wood, C. M., Eibl, M., & Klinck, H. (2021). BirdNET: A deep learning solution for avian diversity monitoring. Ecological Informatics, 61, 101236. [[Source](https://www.sciencedirect.com/science/article/pii/S1574954121000273)]

Doshi, Ketan. (2021). Audio Deep Learning Made Simple: Sound Classification, Step-by-Step. Medium. [[Source](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5.)]

An example of a sound classification implementation using several tools mentioned in the Audax Jax framework, namely pytorch, numpy, Melspectrograms, etc.

**What data will you use? If you are collecting new data, how will you do it?**

Cornell Lab of Ornithology has prepared sample data in the kaggle competition known as BirdCLEF 2022. This data will be the source of training and evaluating our model.

[https://www.kaggle.com/competitions/birdclef-2022](https://www.kaggle.com/competitions/birdclef-2022)

**What method or algorithm are you proposing? If there are existing implementations, will you use them and how? How do you plan to improve or modify such implementations? You don’t have to have an exact answer at this point, but you should have a general sense of how you will approach the problem you are working on.**

Currently, we are interested in exploring ensemble learning and techniques to train the model.

**How will you evaluate your results? Qualitatively, what kind of results do you expect (e.g. plots or figures)? Quantitatively, what kind of analysis will you use to evaluate and/or compare your results (e.g. what performance metrics or statistical tests)?**

Results will be evaluated through the kaggle competition’s chosen metric. In BirdClef 2022, the metric is an F-score metric that will compare the model’s ability to correctly determine specified calls within the soundscape samples.  
  
The traditional F-measure or balanced F-score (F1 score) is the [harmonic mean](https://en.wikipedia.org/wiki/Harmonic_mean#Harmonic_mean_of_two_numbers) of precision and recall.

![{\displaystyle F_{1}={\frac {2}{\mathrm {recall^{-1}} +\mathrm {precision^{-1}} }}=2\cdot {\frac {\mathrm {precision} \cdot \mathrm {recall} }{\mathrm {precision} +\mathrm {recall} }}={\frac {\mathrm {tp} }{\mathrm {tp} +{\frac {1}{2}}(\mathrm {fp} +\mathrm {fn} )}}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/4179c69cf1dde8418c4593177521847e862e7df8).


