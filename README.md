
This is the version 2.0 of PRAGATI. To find the earlier work, please visit branch "PRAGATI-legacy".

## Installation and Running instructions

1. Clone the repository on to your local machine: `git clone https://github.com/ShuvraneelMitra/PRAGATI.git`
2. Navigate into the `PRAGATI` folder: `cd PRAGATI`
3. Create a virtual environment in your directory: `python -m venv venv`
4. Now activate the virtual environment using `source venv/bin/activate` for Linux or (either `.\venv\Scripts\activate.
   ps1` or `.\venv\Scripts\activate.bat` for Windows
5. Sync with the `requirements.txt` file inside your `venv`: `pip install -r requirements.txt`. This will install all 
   the 
   relevant packages inside your virtual environment.
6. Install locally, an older version of timm: `pip install timm==0.5.4 -t old_pkgs/timm0.5.4`
7. Run the app with `uvicorn ui:app --reload --port 8080`. You can choose any port on the localhost of your liking 
   and change it if that port turns out to be blocked.
8. Open `http://127.0.0.1:<port-number>` to get the development server running.

