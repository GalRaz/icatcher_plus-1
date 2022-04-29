### Best Results (test set)
<table>
        <tr>
                <td align="center"> <img src="https://github.com/yoterel/icatcher_plus/blob/main/assets/agreement.png"  alt="0" width = 400px height = 300px ></td>
                <td align="center"><img src="https://github.com/yoterel/icatcher_plus/blob/main/assets/agreement_vs_confidence.png"  alt="0" width = 400px height = 300px ></td>
        </tr>
        <tr><td colspan=2>California-BW</td></tr>
        <tr>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/main/assets/cali-bw_bar.png" alt="0" width = 400px height = 300px></td>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/main/assets/cali-bw_conf.png" alt="1" width = 300px height = 300px></td>
        </tr>
        <tr><td colspan=2>Lookit</td></tr>
        <tr>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/main/assets/lookit_bar.png" alt="0" width = 400px height = 300px></td>
                <td><img src="https://github.com/yoterel/icatcher_plus/blob/main/assets/lookit_conf.png" alt="1" width = 300px height = 300px></td>
        </tr>
</table>

### Installation
Use conda with the environment.yml file to create a virtual environment:

`conda env create --prefix /path/to/virtual/environment -f "/path/to/environment.yml"`

another system requirement is having [ffmpeg](https://ffmpeg.org/download.html) installed.

### Project Structure:


    ├── assets                  # contains assets for README
    ├── datasets                # place holder for datasets 
    ├── face_classifier         # contains all specific code for face classification, separated from main project files on purpose.
        ├── fc_data.py          # creates face classifier dataset
        ├── fc_eval.py          # face classifier evaluation
        ├── fc_model.py         # face classifier model
        ├── fc_train.py         # face classifier training  script
    ├── models                  # place holder for model files
    ├── plots                   # place holder for various plots
    ├── statistics              # code for analyzing multi-variant video dataset statistics
    ├── tests                   # pytests
    ├── augmentations.py        # defines RandAugment set of augmentations
    ├── data.py                 # dataloaders and datasets
    ├── logger.py               # simple logger class
    ├── models.py               # definition of model architectures etc
    ├── options.py              # parse command line arguments
    ├── parsers.py              # annotations (labels) of videos in various formats are parsed using classes in this file
    ├── preprocess.py           # used to parse a raw dataset from OSF into a dataloader ready dataset
    ├── test.py                 # use this to run a full test on a video or a folder of videos
    ├── train.py                # main training loop
    ├── video.py                # API to ffmpeg functionallity
    ├── visualize.py            # compares human annotation with results from icatcher to reproduce paper results
    
### Contributions
Feel free to contribute by submitting a pull request. Make sure to run all tests under /tests


