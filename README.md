# Bachelor Thesis

One Paragraph of project description goes here

## Getting Started

These instructions will get you a copy of the project up and running on your local machine. Make sure that you have downloaded the ScanNet dataset or have at least already preprocessed scene files at hand.

### Prerequisites

Overall Structure

Data folder: Contains everthing related to the dataset & preprocessing
    - dataset.py: 
    - preprocessing folder: everything needed to do the preprocessing steps
    - scenes: contains a subfolder with the whole scannet. Furthermore the preprocessed files for segmentation as well as the generates reconstruct GT are in this folder.

Reconstruction folder: Contains everything related to reconstruction
    -model: 
    -models_collection:
    -results: 

    Files: 
    train_scannet.py
    reconstruct.py
    utils.py


Segmentation folder:
    model:
    models_collection
    results:
    results_collection

    Files.:
    

```
Give examples
```

### Project Structure


Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Fe