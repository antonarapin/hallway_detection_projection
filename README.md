# Hallway detection and image projection on the wall

### We use Hough Transform to detect straight lines. Then we filter the lines and find the vanishing point of the hallway. We separate lines that signify walls, floor and ceiling, and project the image on the extensions of the linese of a given plane finding the homography transformation.

### we worked together with [Zhuofan Tang](https://github.com/tangticco)
(I lost the original repo somewhere so have to reupload this one)


To install all dependencies, run: `pip install -r requirements.txt`

After all necessary packages are installed, run the project code using the master script :

`python projection_master.py <base image filename> <projection image filename>`

An example would be : `python projection_master.py hall4c.jpg earth.jpg`


PS: The best way to run our project would be under a virtual environment (virtualenv).
The way to install it is simply typing sudo pip install virtualenv.
And to create the virtual environment, the user can type : virtualenv --python=/usr/bin/python2.7 <testing directory name>
Then to activate the virtual environment, the user can type: source bin/activate
