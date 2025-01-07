import pathway as pw
from dotenv import load_dotenv
import os

load_dotenv()

table = pw.io.gdrive.read(
    object_id=os.environ.get("OBJECT_ID"),
    service_user_credentials_file="credentials.json"
)
print(dir(table))
pw.run()