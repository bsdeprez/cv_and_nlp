# Tracking passenger movement using descriptive identifications
## Introduction
Most trains are equipped with Closed Circuit Television (CCTV) cameras, which provide valuable data to determine the occupancy of train cars. This information can facilitate the allocation of passengers to less crowded cars, while directing them away from congested ones. However, monitoring the movement of passengers from one car, and consequently one CCTV feed, to another poses a challenge. This project aims to explore the feasibility of recognizing individuals on a CCTV feed and generating a description of each person that is easily understandable to humans, such as "A person wearing a blue sweater with short, black hair". Subsequently, we aim to develop a Natural Language Processing (NLP) model that can distinguish between descriptions referring to the same person and those pertaining to different individuals. In this manner, we can track people in a way that is less invasive to their privacy, without retaining pictures or embeddings of individuals. Furthermore, it creates an explainable tracking mechanism.

## Data availability
We have acquired access to video footage of individuals moving inside a train car, obtained during a prior student project conducted on behalf of Televic Rail. This dataset contains both images and videos, similar to the example presented here.

This dataset will be used as test set, but it still needs to be annotated.

