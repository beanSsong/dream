# Dream Team collaboration for the DREAM Olfaction Challenge

This is all the code you need to reproduce my predictions.
The basic ideas are explained in the wiki, but you can follow step-by-step in the IPython notebooks [dream1.ipynb](http://nbviewer.ipython.org/github/rgerkin/dream/blob/master/dream1.ipynb) and [dream2.ipynb](http://nbviewer.ipython.org/github/rgerkin/dream/blob/master/dream1.ipynb).
Note that in both files the write-to-disk flag is turned off, and the number of estimators (determing prediction quality) is set low to increase speed for demonstration.
These can be changed near the bottom of each file, setting write=True, and n_estimators=1000 (for example).
These notebooks call code in dream.py, loading.py, scoring.py, fit1.py, and fit2.py.
I also show some exploration of parameters and different approaches in [exploration.ipynb](nbviewer.ipython.org/github/rgerkin/dream/blob/master/dream1.ipynb), although this notebook is still a work in progress.  
