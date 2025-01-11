"""
This file contains functions to get the object IDs for
all the files provided in the given GDrive folder containing
the papers and references.
"""

import requests
import os
import json
from dotenv import load_dotenv
from pprint import pprint

def get_files(folder_id, api_key):
        url = "https://www.googleapis.com/drive/v3/files"
        params = {
            "q": f"'{folder_id}' in parents",
            "key": api_key,
            "fields": "files(id, name, mimeType, webViewLink)"
        }
        response = requests.get(url, params=params)

        if response.status_code == 200:
            return response.json().get("files", [])
        else:
            pprint(f"Error: {response.status_code}")
            pprint(response.json())
            return []
        

def traverse_folder(folder_id, files_list, api_key):
    items = get_files(folder_id, api_key)
    for item in items:
        if item["mimeType"] == "application/vnd.google-apps.folder":
            traverse_folder(item["id"], files_list, api_key)
        elif item["name"][0] == "R":
            files_list["Reference"].append({
                "Name": item["name"],
                "url": item["webViewLink"]
            }) 
        else:
            files_list["Papers"].append({
                "Name": item["name"],
                "Object_ID": item["webViewLink"].split('/')[-2]
            })


def get_all_files(folder_id, api_key):
    files_list = {"Papers": [], "Reference": []}
    traverse_folder(folder_id, files_list, api_key)
    return files_list

def main():

    FOLDER_ID = os.getenv("OVERALL_OBJECT_ID")
    API_KEY = os.getenv("GDRIVE_API_KEY")

    files = get_all_files(FOLDER_ID, API_KEY)
    if files:
        with open("utils/fileIDs/Papers.json", 'w') as f:
            sorted_f = sorted(files["Papers"], key=lambda x: x["Name"])
            json.dump(sorted_f, f, indent=4)
        with open("utils/fileIDs/Reference.json", 'w') as f:
            sorted_f = sorted(files["Reference"], key=lambda x: x["Name"])
            json.dump(sorted_f, f, indent=4)
        print("Object IDs written to the appropriate files! \nPlease check the utils/ folder in your workspace")
    else:
        print("No files found.")


if __name__ == "__main__":
    main()

