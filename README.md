# ChessRecognition

## Setup
- install requirements from file
- create debug folder in the project


## Find the corners
- There are two  files that search for corners in a given picture
- Get Slid for [Maciej A. Czyzewski's Algorithm](https://arxiv.org/abs/1708.03898)
    - Usage eg: python3 get_slid.py data/chessboards/1.jpg
    - prints corner points, saves cropped board and a corner image in debug folder
- Get Points for my implementation ( not finished)
    - Usage eg:  python3 get_points.py data/chessboards/1.jpg
    - prints corner points, saves cropped board and a corner image in debug folder
