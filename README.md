# Spotify Music Label Song Recommender
### By Justin Williams and Khyatee Desai

## Overview

The goal of this project was to build a Spotify recommendation engine for music record labels. The recommender takes in a users playlist, and generates a playlist with like songs from a specific record label. This enables the record label to showcase artists songs that most closely align with a particular users listening habits, thus providing each listener with a personalized playlist. Data used for this project was gathered from a variety of sources: 1) Record label artist roster data was sourced through web-scraping, 2) Spotify song database was obtained through Kaggle, 3) Addtional information was gathered directly through the Spotify API. The methodologies utilized included data cleaning, outlier impution, exploratory data analysis (EDA) and utilization of both cosine similarity and KMeans clusertering to create two distinct recommenders. Several Flatirion students and staff members submitted their playlists and recommendations were generated. Participants were then asked to rate each plalist. The results indicated that _______. 

## Business Problem

Many music record labels need to get exposure for their artists. However, with the plethora of music available through streaming, it has become difficult for record labels to obtain new listeners. That said, wouldn't it be ideal if a record label could take a prospective listeners playlist and provide them with a personalized one composed entirely of songs from their artist roster? We thought so, that's why we created this music label song recommender. Subsequentially, this enables prospective listeners to get a preview of a record labels product, specifically taylored to their favorite songs. This provides them with a personalized introduction to the record labels artist roster. 

## Data

The data used in this project was obtained from a few different sources. Record label roster data was scraped directly from each respective labels website. A dataset gathered from Spotify was downloaded from kaggle, and supplemented with information streamed directly from the Spotify API. The Spotify data contains audio features for each track, they are as follows:

| KEY              | VALUE TYPE | VALUE DESCRIPTION                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|------------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| duration_ms      | int        | The duration of the track in milliseconds.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| key              | int        | The estimated overall key of the track. Integers map to pitches using standard Pitch Class notation . E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1.                                                                                                                                                                                                                                                                                                                                                                                         |
| mode             | int        | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.                                                                                                                                                                                                                                                                                                                                                                                                                 |
| time_signature   | int        | An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).                                                                                                                                                                                                                                                                                                                                                                                                                   |
| acousticness     | float      | A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. The distribution of values for this feature look like this:#                                                                                                                                                                                                                                                                                                                                                                                       |
| danceability     | float      | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. The distribution of values for this feature look like this:#                                                                                                                                                                                                                                                                       |
| energy           | float      | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. The distribution of values for this feature look like this:#                                                                                                                          |
| instrumentalness | float      | Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. The distribution of values for this feature look like this:#                                                                                                                 |
| liveness         | float      | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. The distribution of values for this feature look like this:#                                                                                                                                                                                                                                                                                            |
| loudness         | float      | The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typical range between -60 and 0 db. The distribution of values for this feature look like this:#                                                                                                                                                                                       |
| speechiness      | float      | Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. The distribution of values for this feature look like this:# |
| valence          | float      | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). The distribution of values for this feature look like this:#                                                                                                                                                                                                                                                                  |
| tempo            | float      | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. The distribution of values for this feature look like this:#                                                                                                                                                                                                                                                                                                                         |
| id               | string     | The Spotify ID for the track.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| uri              | string     | The Spotify URI for the track.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| track_href       | string     | A link to the Web API endpoint providing full details of the track.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| analysis_url     | string     | An HTTP URL to access the full audio analysis of this track. An access token is required to access this data.                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| type             | string     | The object type: “audio_features”                                                                                                                                                                                                                                           

## Methods

Describe the process for analyzing or modeling the data. For Phase 1, this will be descriptive analysis.

The aforementioned song attributes from the Spotify API Audio Features Object, enabled us to caculate similarities between specific music record labels catalogs and users inputted playlists. Before caculating similarity metrics, data was preprocessed through the following pipline: 
- Dropping of unecessary features
- Feature engineering i.e. decade imputation, key/mode concatenation
- Outliers bought down to 5 standard deviations from mean
- Unscaled features scaled
- Dummy columns created for categorical
- Scraped record label data merged with Spotify data frame

Once preprocessing steps were complete, recommendations were generated with two distinct techniques:
1) **Cosine similarity** - Caculating cosine similarities between record label songs and users playlist. 
2) **KMeans Clustering** - Clustering record labels songs into k clusters and mapping users playlist to the most similiar cluster.

## Results

TBD

### Song distribution by year, by label
![visual1](./images/label_year_song_hist.png)

### Key/Mode by popularity
![visual2](./images/key_mode_pop_bar.png)

### Song attribute by label
![visual3](./images/attr_label_plot.png)

### Most variant song attributes
![visual4](./images/heatmap_variant_attr.png)


## Conclusions

TBD

## For More Information

Please review our full analysis in [our Jupyter Notebooks](./notebooks) or our [presentation](./music_recommender_project_presentation.pdf).

For any additional questions, please contact **Justin Williams - justinmorganwilliams@newschool.edu and Khyatee Desai - khyatee.d@gmail.com**

## Repository Structure

Describe the structure of your repository and its contents, for example:

```
├── README.md                                    <- Top-level README for reviewers of this project
├── data_collection_and_eda.ipynb                <- Narrative documentation of analysis in Jupyter notebook
├── modeling.ipynb                               <- Modeling and recommender in Jupyter notebook
├── music_recommender_project_presentation.pdf   <- PDF version of project presentation
├── data                                         <- Both sourced externally and generated from code
└── images                                       <- Both sourced externally and generated from code
```
