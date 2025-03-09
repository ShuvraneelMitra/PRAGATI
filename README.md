
This is the version 2.0 of PRAGATI. To find the earlier work, please visit branch "PRAGATI-legacy".

## Installation and Running instructions

1. Clone the repository on to your local machine: `git clone https://github.com/ShuvraneelMitra/PRAGATI.git`
2. Navigate into the `PRAGATI` folder: `cd PRAGATI`
3. Create a virtual environment in your directory: `python -m venv venv`
4. Now activate the virtual environment using `source venv/bin/activate` for Linux or `.\venv\Scripts\activate.
   ps1` for Windows
5. Install `pip-tools`: `pip install pip-tools`
6. Sync with the `requirements.txt` file inside your `venv`: `pip-sync requirements.txt`. This will install all the 
   relevant packages inside your virtual environment.
7. Run the app with `uvicorn ui:app --reload --port 8080`. You can choose any port on the localhost of your liking 
   and change it if that port turns out to be blocked.
8. Open `http://127.0.0.1:<port-number>` to get the development server running.

