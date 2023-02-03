
---

# Project Summary

    This is a framework for efficiently backtesting option spreads (Call/Put Credit spreads) performance. You can customize different spread parameters to backtest your strategy with SPY data for the past 10 years.

# Install 
    1. Create and activate a new Anaconda environment

    2. In anaconda power shell, run
   
        !pip3 install -r requirements.txt

        (tips: if you are having trouble installing ta-lib, please go to https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib to download and install wheel package specific to your python version and operating system)
   
    3. Download the preprocessed spread data from: 
   
        [Google Drive](https://drive.google.com/drive/folders/1osSFFwno6Uk-Z3hMDrxyix5dbdbFHgMI?usp=sharing) 
        
        and unzip the file under Option_Spreads_Backtesting/Spreads_Data/SPY folder.

    4. See main.ipynb for futher instructions.

# To-Do
    1. Implement a customizable technical indicator method
    2. Create a statistic report method